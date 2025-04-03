"""
백테스트 엔진 모듈

이 모듈은 과거 시장 데이터를 기반으로 전략을 테스트합니다.
주요 기능:
- 과거 데이터 기반 전략 테스트
- 다양한 시장 상황 시뮬레이션
- 성과 지표 계산
- 결과 시각화 및 보고서 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from ..strategies.base import Strategy
from ..utils.logger import setup_logger
from ..strategies.integrated_strategy import IntegratedStrategy
from ..trading.risk_manager import RiskManager
from ..trading.position_sizing import PositionSizing
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from src.trading.strategy import IntegratedStrategy
from src.backtest.visualization import BacktestVisualizer

from src.trading.strategy import IntegratedStrategy
from src.backtest.data import BacktestData
from src.trading.simulation import TradingSimulator, SimulationConfig

@dataclass
class BacktestMetrics:
    total_return: float  # 총 수익률
    annual_return: float  # 연간 수익률
    sharpe_ratio: float  # 샤프 비율
    calmar_ratio: float  # 칼마 비율
    mar_ratio: float  # MAR 비율
    max_drawdown: float  # 최대 낙폭
    win_rate: float  # 승률
    profit_factor: float  # 수익 팩터
    max_consecutive_losses: int  # 최대 연속 손실
    avg_trade_duration: timedelta  # 평균 거래 기간
    monthly_returns: pd.Series  # 월별 수익률
    quarterly_returns: pd.Series  # 분기별 수익률
    market_condition_returns: Dict[str, float]  # 시장 상황별 수익률

@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: List[Dict]
    equity_curve: pd.Series
    position_sizes: pd.Series
    leverage_history: pd.Series
    market_conditions: pd.Series

class Backtester:
    """
    전략 백테스트를 수행하는 클래스
    """
    
    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.001,    # 0.1%
        data_dir: str = 'data/backtest',
        results_dir: str = 'results/backtest'
    ):
        """
        백테스터 초기화
        
        Args:
            strategy (Strategy): 테스트할 전략
            initial_capital (float): 초기 자본금
            commission (float): 거래 수수료
            slippage (float): 슬리피지
            data_dir (str): 데이터 저장 디렉토리
            results_dir (str): 결과 저장 디렉토리
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # 상태 변수 초기화
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        
        # 로거 설정
        self.logger = setup_logger()
        self.logger.info("Backtester initialized")
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            start_date (datetime): 시작 날짜
            end_date (datetime): 종료 날짜
            
        Returns:
            Dict[str, Any]: 백테스트 결과
        """
        try:
            self.logger.info("Starting backtest")
            
            # 데이터 필터링
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # 전략 초기화
            self.strategy.initialize(data)
            
            # 백테스트 실행
            for i in range(len(data)):
                current_data = data.iloc[:i+1]
                current_time = data.index[i]
                current_price = data.iloc[i]['close']
                
                # 포지션 업데이트
                self._update_positions(current_data, i, current_price, current_time)
                
                # 새로운 포지션 진입 확인
                if len(self.positions) < self.strategy.max_positions:
                    self._check_entry_signals(current_data, i, current_price, current_time)
                
                # 자본금 곡선 업데이트
                self._update_equity_curve(current_price)
            
            # 결과 생성
            results = self._generate_results(data)
            
            # 결과 저장
            self._save_results(results)
            
            self.logger.info("Backtest completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _update_positions(
        self,
        data: pd.DataFrame,
        index: int,
        current_price: float,
        current_time: datetime
    ) -> None:
        """
        포지션 상태 업데이트
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            index (int): 현재 데이터 인덱스
            current_price (float): 현재 가격
            current_time (datetime): 현재 시간
        """
        try:
            # 각 포지션에 대해 청산 조건 확인
            for position_id, position in list(self.positions.items()):
                if self.strategy.should_exit(data, index, position):
                    # 청산 가격 계산 (슬리피지 고려)
                    if position['position_type'] == 'long':
                        exit_price = current_price * (1 - self.slippage)
                    else:
                        exit_price = current_price * (1 + self.slippage)
                    
                    # 포지션 청산
                    self._close_position(
                        position_id=position_id,
                        exit_price=exit_price,
                        exit_time=current_time,
                        exit_reason='strategy_signal'
                    )
                
                else:
                    # 포지션 PnL 업데이트
                    if position['position_type'] == 'long':
                        pnl = (current_price - position['entry_price']) * position['position_size']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['position_size']
                    
                    position['current_price'] = current_price
                    position['pnl'] = pnl
                    
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
            raise
    
    def _check_entry_signals(
        self,
        data: pd.DataFrame,
        index: int,
        current_price: float,
        current_time: datetime
    ) -> None:
        """
        진입 신호 확인 및 포지션 진입
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            index (int): 현재 데이터 인덱스
            current_price (float): 현재 가격
            current_time (datetime): 현재 시간
        """
        try:
            # 롱 진입 신호 확인
            if self.strategy.should_long(data, index):
                # 진입 가격 계산 (슬리피지 고려)
                entry_price = current_price * (1 + self.slippage)
                
                # 포지션 크기 계산
                position_size = self.strategy.calculate_position_size(entry_price)
                
                # 수수료 계산
                commission_cost = entry_price * position_size * self.commission
                
                # 포지션 진입
                position_id = f"long_{current_time.strftime('%Y%m%d_%H%M%S')}"
                self._open_position(
                    position_id=position_id,
                    entry_price=entry_price,
                    position_size=position_size,
                    position_type='long',
                    timestamp=current_time,
                    commission=commission_cost
                )
            
            # 숏 진입 신호 확인
            elif self.strategy.should_short(data, index):
                # 진입 가격 계산 (슬리피지 고려)
                entry_price = current_price * (1 - self.slippage)
                
                # 포지션 크기 계산
                position_size = self.strategy.calculate_position_size(entry_price)
                
                # 수수료 계산
                commission_cost = entry_price * position_size * self.commission
                
                # 포지션 진입
                position_id = f"short_{current_time.strftime('%Y%m%d_%H%M%S')}"
                self._open_position(
                    position_id=position_id,
                    entry_price=entry_price,
                    position_size=position_size,
                    position_type='short',
                    timestamp=current_time,
                    commission=commission_cost
                )
                
        except Exception as e:
            self.logger.error(f"Error checking entry signals: {str(e)}")
            raise
    
    def _open_position(
        self,
        position_id: str,
        entry_price: float,
        position_size: float,
        position_type: str,
        timestamp: datetime,
        commission: float
    ) -> None:
        """
        포지션 진입
        
        Args:
            position_id (str): 포지션 ID
            entry_price (float): 진입 가격
            position_size (float): 포지션 크기
            position_type (str): 포지션 타입
            timestamp (datetime): 진입 시간
            commission (float): 수수료
        """
        try:
            # 포지션 정보 저장
            self.positions[position_id] = {
                'entry_price': entry_price,
                'position_size': position_size,
                'position_type': position_type,
                'entry_time': timestamp,
                'current_price': entry_price,
                'pnl': 0.0,
                'commission': commission
            }
            
            # 자본금에서 수수료 차감
            self.current_capital -= commission
            
            self.logger.info(
                f"Position opened: {position_id} - {position_type} {position_size} "
                f"@ {entry_price} (commission: {commission:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            raise
    
    def _close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> None:
        """
        포지션 청산
        
        Args:
            position_id (str): 포지션 ID
            exit_price (float): 청산 가격
            exit_time (datetime): 청산 시간
            exit_reason (str): 청산 이유
        """
        try:
            if position_id not in self.positions:
                raise ValueError(f"Position {position_id} not found")
            
            position = self.positions[position_id]
            
            # PnL 계산
            if position['position_type'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['position_size']
            else:
                pnl = (position['entry_price'] - exit_price) * position['position_size']
            
            # 청산 수수료 계산
            exit_commission = exit_price * position['position_size'] * self.commission
            
            # 순 PnL 계산 (수수료 제외)
            net_pnl = pnl - position['commission'] - exit_commission
            
            # 자본금 업데이트
            self.current_capital += net_pnl
            
            # 거래 기록 저장
            trade = {
                'position_id': position_id,
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'position_size': position['position_size'],
                'position_type': position['position_type'],
                'pnl': net_pnl,
                'commission': position['commission'] + exit_commission,
                'exit_reason': exit_reason
            }
            self.trades.append(trade)
            
            # 포지션 제거
            del self.positions[position_id]
            
            self.logger.info(
                f"Position closed: {position_id} - {position['position_type']} "
                f"{position['position_size']} @ {exit_price} (PnL: {net_pnl:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise
    
    def _update_equity_curve(self, current_price: float) -> None:
        """
        자본금 곡선 업데이트
        
        Args:
            current_price (float): 현재 가격
        """
        try:
            # 포지션 PnL 계산
            total_pnl = sum(position['pnl'] for position in self.positions.values())
            
            # 현재 자본금 계산
            current_equity = self.current_capital + total_pnl
            
            # 자본금 곡선 업데이트
            self.equity_curve.append(current_equity)
            
        except Exception as e:
            self.logger.error(f"Error updating equity curve: {str(e)}")
            raise
    
    def _generate_results(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        백테스트 결과 생성
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Dict[str, Any]: 백테스트 결과
        """
        try:
            # 기본 통계
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 수익률 계산
            total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            
            # 평균 수익/손실
            profits = [t['pnl'] for t in self.trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # 최대 낙폭 (MDD)
            equity_curve = pd.Series(self.equity_curve)
            rolling_max = equity_curve.expanding().max()
            drawdowns = (rolling_max - equity_curve) / rolling_max * 100
            max_drawdown = drawdowns.max()
            
            # 샤프 비율
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 0 else 0
            
            # 결과 생성
            results = {
                'summary': {
                    'initial_capital': self.initial_capital,
                    'final_capital': self.current_capital,
                    'total_return': total_return,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'total_commission': sum(t['commission'] for t in self.trades)
                },
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'parameters': {
                    'commission': self.commission,
                    'slippage': self.slippage,
                    'strategy_params': self.strategy.config
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating results: {str(e)}")
            raise
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        백테스트 결과 저장
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
        """
        try:
            # 결과 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_results_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            # 결과 저장
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            self.logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

class BacktestEngine:
    """
    백테스트 엔진 클래스
    """
    
    def __init__(self, strategy: IntegratedStrategy, config: Dict[str, Any]):
        """
        백테스트 엔진 초기화
        
        Args:
            strategy (IntegratedStrategy): 거래 전략
            config (Dict[str, Any]): 설정 정보
        """
        self.strategy = strategy
        self.config = config
        self.logger = setup_logger()
        
        # 백테스트 상태
        self.is_running = False
        self.current_capital = config['backtest']['initial_capital']
        self.positions = {}
        self.trades = []
        self.equity_curve = pd.Series()
        
        # 시각화기 초기화
        self.visualizer = BacktestVisualizer()
        
    async def initialize(self):
        """백테스트 엔진 초기화"""
        try:
            self.logger.info("백테스트 엔진 초기화")
            await self.strategy.initialize()
        except Exception as e:
            self.logger.error(f"백테스트 엔진 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """백테스트 엔진 종료"""
        try:
            self.logger.info("백테스트 엔진 종료")
            await self.strategy.close()
        except Exception as e:
            self.logger.error(f"백테스트 엔진 종료 실패: {str(e)}")
            raise
            
    async def run(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간대
            start_date (datetime): 시작 날짜
            end_date (datetime): 종료 날짜
            
        Returns:
            Dict[str, Any]: 백테스트 결과
        """
        try:
            self.is_running = True
            self.logger.info(f"백테스트 시작: {symbol} {timeframe}")
            
            # 초기 자본금 설정
            self.current_capital = self.config['backtest']['initial_capital']
            self.positions = {}
            self.trades = []
            self.equity_curve = pd.Series()
            
            # 시장 데이터 수집
            market_data = await self.strategy.get_market_data(
                symbol,
                timeframe,
                start_date,
                end_date
            )
            
            # 백테스트 실행
            for i in range(len(market_data)):
                if not self.is_running:
                    break
                    
                current_data = market_data.iloc[i]
                
                # 거래 신호 생성
                signal = await self.strategy.generate_signal(current_data)
                
                if signal:
                    # 주문 실행
                    await self._execute_trade(signal, current_data)
                    
                # 포지션 모니터링
                await self._monitor_positions(current_data)
                
                # 자본금 곡선 업데이트
                self._update_equity_curve(current_data.name)
                
            # 성과 지표 계산
            results = self._calculate_metrics()
            
            # 결과 시각화
            report_file = self.visualizer.generate_report(
                results,
                symbol,
                timeframe
            )
            
            self.logger.info(f"백테스트 완료: {report_file}")
            return results
            
        except Exception as e:
            self.logger.error(f"백테스트 실행 실패: {str(e)}")
            raise
        finally:
            self.is_running = False
            
    async def _execute_trade(self, signal: Dict[str, Any], market_data: pd.Series):
        """
        거래 실행
        
        Args:
            signal (Dict[str, Any]): 거래 신호
            market_data (pd.Series): 시장 데이터
        """
        try:
            # 포지션 크기 계산
            size = self._calculate_position_size(signal)
            
            # 주문 실행
            if signal['side'] == 'buy':
                if signal['symbol'] in self.positions:
                    # 포지션 청산
                    await self._close_position(
                        signal['symbol'],
                        market_data['close'],
                        'close'
                    )
                # 롱 포지션 진입
                self.positions[signal['symbol']] = {
                    'side': 'buy',
                    'size': size,
                    'entry_price': market_data['close'],
                    'entry_time': market_data.name
                }
            else:
                if signal['symbol'] in self.positions:
                    # 포지션 청산
                    await self._close_position(
                        signal['symbol'],
                        market_data['close'],
                        'close'
                    )
                # 숏 포지션 진입
                self.positions[signal['symbol']] = {
                    'side': 'sell',
                    'size': size,
                    'entry_price': market_data['close'],
                    'entry_time': market_data.name
                }
                
        except Exception as e:
            self.logger.error(f"거래 실행 실패: {str(e)}")
            raise
            
    async def _monitor_positions(self, market_data: pd.Series):
        """
        포지션 모니터링
        
        Args:
            market_data (pd.Series): 시장 데이터
        """
        try:
            for symbol, position in list(self.positions.items()):
                # 손익 계산
                pnl = self._calculate_pnl(position, market_data['close'])
                
                # 손절/익절 확인
                if pnl <= -self.config['trading']['stop_loss']:
                    await self._close_position(
                        symbol,
                        market_data['close'],
                        'stop_loss'
                    )
                elif pnl >= self.config['trading']['take_profit']:
                    await self._close_position(
                        symbol,
                        market_data['close'],
                        'take_profit'
                    )
                    
        except Exception as e:
            self.logger.error(f"포지션 모니터링 실패: {str(e)}")
            raise
            
    async def _close_position(
        self,
        symbol: str,
        price: float,
        reason: str
    ):
        """
        포지션 청산
        
        Args:
            symbol (str): 거래 심볼
            price (float): 청산 가격
            reason (str): 청산 사유
        """
        try:
            position = self.positions[symbol]
            
            # 손익 계산
            pnl = self._calculate_pnl(position, price)
            
            # 거래 기록 추가
            self.trades.append({
                'symbol': symbol,
                'side': position['side'],
                'size': position['size'],
                'entry_price': position['entry_price'],
                'exit_price': price,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'pnl': pnl,
                'reason': reason
            })
            
            # 자본금 업데이트
            self.current_capital += pnl
            
            # 포지션 제거
            del self.positions[symbol]
            
        except Exception as e:
            self.logger.error(f"포지션 청산 실패: {str(e)}")
            raise
            
    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal (Dict[str, Any]): 거래 신호
            
        Returns:
            float: 포지션 크기
        """
        try:
            # 리스크 관리 설정
            risk_per_trade = self.config['trading']['risk_per_trade']
            stop_loss = self.config['trading']['stop_loss']
            
            # 포지션 크기 계산
            size = self.current_capital * risk_per_trade / stop_loss
            
            return size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            raise
            
    def _calculate_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        """
        손익 계산
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            current_price (float): 현재 가격
            
        Returns:
            float: 손익
        """
        try:
            if position['side'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
                
            return pnl
            
        except Exception as e:
            self.logger.error(f"손익 계산 실패: {str(e)}")
            raise
            
    def _update_equity_curve(self, timestamp: datetime):
        """
        자본금 곡선 업데이트
        
        Args:
            timestamp (datetime): 시간
        """
        try:
            # 미실현 손익 계산
            unrealized_pnl = 0
            for position in self.positions.values():
                unrealized_pnl += self._calculate_pnl(
                    position,
                    position['entry_price']
                )
                
            # 자본금 곡선 업데이트
            self.equity_curve[timestamp] = self.current_capital + unrealized_pnl
            
        except Exception as e:
            self.logger.error(f"자본금 곡선 업데이트 실패: {str(e)}")
            raise
            
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        성과 지표 계산
        
        Returns:
            Dict[str, Any]: 성과 지표
        """
        try:
            # 기본 지표
            total_return = (self.current_capital - self.config['backtest']['initial_capital']) / self.config['backtest']['initial_capital']
            total_trades = len(self.trades)
            win_trades = len([t for t in self.trades if t['pnl'] > 0])
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            # 수익률 계산
            returns = pd.Series([t['pnl'] / self.config['backtest']['initial_capital'] for t in self.trades])
            avg_return = returns.mean() if len(returns) > 0 else 0
            return_std = returns.std() if len(returns) > 1 else 0
            
            # 샤프 비율 계산
            risk_free_rate = 0.02  # 연간 2% 가정
            excess_returns = returns - risk_free_rate/252  # 일간 무위험 수익률
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 1 else 0
            
            # 최대 낙폭 계산
            rolling_max = self.equity_curve.expanding().max()
            drawdown = (self.equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 결과 반환
            return {
                'total_return': total_return,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'avg_return': avg_return,
                'return_std': return_std,
                'equity_curve': self.equity_curve.to_dict(),
                'trades': self.trades
            }
            
        except Exception as e:
            self.logger.error(f"성과 지표 계산 실패: {str(e)}")
            raise
            
    def get_results(self) -> Dict[str, Any]:
        """
        백테스트 결과 조회
        
        Returns:
            Dict[str, Any]: 백테스트 결과
        """
        return self._calculate_metrics() 