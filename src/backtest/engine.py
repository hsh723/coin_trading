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
from ..strategy.base_strategy import BaseStrategy
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
from ..data.collector import DataCollector
from ..data.processor import DataProcessor
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

logger = setup_logger('backtest_engine')

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
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.001,    # 0.1%
        data_dir: str = 'data/backtest',
        results_dir: str = 'results/backtest'
    ):
        """
        백테스터 초기화
        
        Args:
            strategy (BaseStrategy): 테스트할 전략
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
    """백테스팅 엔진"""
    
    def __init__(self,
                 config_path: str = "./config/backtest_config.json",
                 data_dir: str = "./data",
                 log_dir: str = "./logs"):
        """
        백테스팅 엔진 초기화
        
        Args:
            config_path: 설정 파일 경로
            data_dir: 데이터 디렉토리
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("backtest")
        
        # 설정 로드
        self.config = self._load_config()
        
        # 백테스팅 결과
        self.results = {}
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def run(self,
            strategy: Any,
            data: pd.DataFrame,
            start_date: str,
            end_date: str,
            initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """
        백테스팅 실행
        
        Args:
            strategy: 트레이딩 전략
            data: 백테스팅 데이터
            start_date: 시작 날짜
            end_date: 종료 날짜
            initial_capital: 초기 자본금
            
        Returns:
            백테스팅 결과
        """
        try:
            # 데이터 필터링
            mask = (data.index >= start_date) & (data.index <= end_date)
            data = data[mask]
            
            # 포트폴리오 초기화
            portfolio = {
                "cash": initial_capital,
                "positions": {},
                "value": initial_capital,
                "returns": []
            }
            
            # 거래 기록
            trades = []
            
            # 백테스팅 실행
            for i in range(len(data)):
                current_date = data.index[i]
                current_data = data.iloc[i]
                
                # 전략 신호 생성
                signals = strategy.generate_signals(current_data)
                
                # 포지션 관리
                self._manage_positions(
                    portfolio=portfolio,
                    signals=signals,
                    current_data=current_data,
                    current_date=current_date,
                    trades=trades
                )
                
                # 포트폴리오 가치 업데이트
                self._update_portfolio_value(portfolio, current_data)
                
                # 수익률 계산
                if i > 0:
                    prev_value = portfolio["value"] / (1 + portfolio["returns"][-1])
                    current_return = (portfolio["value"] - prev_value) / prev_value
                    portfolio["returns"].append(current_return)
                else:
                    portfolio["returns"].append(0.0)
                    
            # 성과 분석
            results = self._analyze_performance(portfolio, trades)
            
            # 결과 저장
            self.results[strategy.name] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"백테스팅 실행 중 오류 발생: {e}")
            raise
            
    def _manage_positions(self,
                         portfolio: Dict[str, Any],
                         signals: Dict[str, Any],
                         current_data: pd.Series,
                         current_date: datetime,
                         trades: List[Dict[str, Any]]) -> None:
        """
        포지션 관리
        
        Args:
            portfolio: 포트폴리오
            signals: 전략 신호
            current_data: 현재 데이터
            current_date: 현재 날짜
            trades: 거래 기록
        """
        try:
            for symbol, signal in signals.items():
                if signal["action"] == "BUY":
                    # 매수 실행
                    price = current_data[f"{symbol}_close"]
                    quantity = signal["quantity"]
                    value = price * quantity
                    
                    if value <= portfolio["cash"]:
                        # 포지션 추가
                        if symbol not in portfolio["positions"]:
                            portfolio["positions"][symbol] = {
                                "quantity": 0,
                                "avg_price": 0,
                                "value": 0
                            }
                            
                        portfolio["positions"][symbol]["quantity"] += quantity
                        portfolio["positions"][symbol]["avg_price"] = (
                            (portfolio["positions"][symbol]["avg_price"] *
                             (portfolio["positions"][symbol]["quantity"] - quantity) +
                             price * quantity) /
                            portfolio["positions"][symbol]["quantity"]
                        )
                        portfolio["positions"][symbol]["value"] = (
                            portfolio["positions"][symbol]["quantity"] * price
                        )
                        
                        # 현금 감소
                        portfolio["cash"] -= value
                        
                        # 거래 기록 추가
                        trades.append({
                            "timestamp": current_date,
                            "symbol": symbol,
                            "action": "BUY",
                            "price": price,
                            "quantity": quantity,
                            "value": value
                        })
                        
                elif signal["action"] == "SELL":
                    # 매도 실행
                    if symbol in portfolio["positions"]:
                        price = current_data[f"{symbol}_close"]
                        quantity = signal["quantity"]
                        value = price * quantity
                        
                        # 포지션 감소
                        portfolio["positions"][symbol]["quantity"] -= quantity
                        portfolio["positions"][symbol]["value"] = (
                            portfolio["positions"][symbol]["quantity"] * price
                        )
                        
                        # 현금 증가
                        portfolio["cash"] += value
                        
                        # 거래 기록 추가
                        trades.append({
                            "timestamp": current_date,
                            "symbol": symbol,
                            "action": "SELL",
                            "price": price,
                            "quantity": quantity,
                            "value": value
                        })
                        
                        # 포지션 제거
                        if portfolio["positions"][symbol]["quantity"] == 0:
                            del portfolio["positions"][symbol]
                            
        except Exception as e:
            self.logger.error(f"포지션 관리 중 오류 발생: {e}")
            raise
            
    def _update_portfolio_value(self,
                               portfolio: Dict[str, Any],
                               current_data: pd.Series) -> None:
        """
        포트폴리오 가치 업데이트
        
        Args:
            portfolio: 포트폴리오
            current_data: 현재 데이터
        """
        try:
            # 포지션 가치 계산
            positions_value = 0.0
            for symbol, position in portfolio["positions"].items():
                price = current_data[f"{symbol}_close"]
                position["value"] = position["quantity"] * price
                positions_value += position["value"]
                
            # 포트폴리오 가치 업데이트
            portfolio["value"] = portfolio["cash"] + positions_value
            
        except Exception as e:
            self.logger.error(f"포트폴리오 가치 업데이트 중 오류 발생: {e}")
            raise
            
    def _analyze_performance(self,
                            portfolio: Dict[str, Any],
                            trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        성과 분석
        
        Args:
            portfolio: 포트폴리오
            trades: 거래 기록
            
        Returns:
            성과 분석 결과
        """
        try:
            # 수익률 계산
            returns = np.array(portfolio["returns"])
            total_return = (portfolio["value"] / portfolio["initial_capital"]) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            # 변동성 계산
            volatility = np.std(returns) * np.sqrt(252)
            
            # 샤프 비율 계산
            risk_free_rate = self.config.get("risk_free_rate", 0.02)
            sharpe_ratio = (annual_return - risk_free_rate) / volatility
            
            # 최대 손실폭 계산
            cum_returns = np.cumprod(1 + returns)
            max_drawdown = np.min(cum_returns / np.maximum.accumulate(cum_returns)) - 1
            
            # 승률 계산
            winning_trades = len([t for t in trades if t["action"] == "SELL" and
                                t["price"] > portfolio["positions"][t["symbol"]]["avg_price"]])
            total_trades = len([t for t in trades if t["action"] == "SELL"])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 평균 수익/손실 계산
            profits = [t["value"] - portfolio["positions"][t["symbol"]]["avg_price"] * t["quantity"]
                      for t in trades if t["action"] == "SELL"]
            avg_profit = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
            avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
            
            # 결과 반환
            return {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "total_trades": total_trades,
                "portfolio_value": portfolio["value"],
                "trades": trades
            }
            
        except Exception as e:
            self.logger.error(f"성과 분석 중 오류 발생: {e}")
            raise
            
    def optimize(self,
                 strategy: Any,
                 data: pd.DataFrame,
                 param_grid: Dict[str, List[Any]],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 1000000.0,
                 n_jobs: int = -1) -> Dict[str, Any]:
        """
        전략 최적화
        
        Args:
            strategy: 트레이딩 전략
            data: 백테스팅 데이터
            param_grid: 파라미터 그리드
            start_date: 시작 날짜
            end_date: 종료 날짜
            initial_capital: 초기 자본금
            n_jobs: 병렬 처리 작업 수
            
        Returns:
            최적화 결과
        """
        try:
            # 병렬 처리 설정
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
                
            # 파라미터 조합 생성
            param_combinations = self._generate_param_combinations(param_grid)
            
            # 병렬 백테스팅 실행
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for params in param_combinations:
                    # 전략 복제
                    strategy_copy = strategy.copy()
                    strategy_copy.set_params(params)
                    
                    # 백테스팅 실행
                    future = executor.submit(
                        self.run,
                        strategy=strategy_copy,
                        data=data,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital
                    )
                    futures.append((params, future))
                    
                # 결과 수집
                results = []
                for params, future in futures:
                    try:
                        result = future.result()
                        result["params"] = params
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"백테스팅 실행 중 오류 발생: {e}")
                        
            # 최적 파라미터 선택
            best_result = max(results, key=lambda x: x["sharpe_ratio"])
            
            return {
                "best_params": best_result["params"],
                "best_performance": best_result,
                "all_results": results
            }
            
        except Exception as e:
            self.logger.error(f"전략 최적화 중 오류 발생: {e}")
            raise
            
    def _generate_param_combinations(self,
                                   param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        파라미터 조합 생성
        
        Args:
            param_grid: 파라미터 그리드
            
        Returns:
            파라미터 조합 리스트
        """
        try:
            # 파라미터 이름과 값 리스트 분리
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # 조합 생성
            from itertools import product
            combinations = list(product(*param_values))
            
            # 딕셔너리로 변환
            param_combinations = []
            for combo in combinations:
                params = dict(zip(param_names, combo))
                param_combinations.append(params)
                
            return param_combinations
            
        except Exception as e:
            self.logger.error(f"파라미터 조합 생성 중 오류 발생: {e}")
            raise
            
    def plot_results(self,
                     results: Dict[str, Any],
                     save_path: Optional[str] = None) -> go.Figure:
        """
        백테스팅 결과 시각화
        
        Args:
            results: 백테스팅 결과
            save_path: 저장 경로
            
        Returns:
            시각화 결과
        """
        try:
            # 포트폴리오 가치 차트
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            
            # 포트폴리오 가치
            fig.add_trace(
                go.Scatter(
                    x=results["dates"],
                    y=results["portfolio_values"],
                    mode="lines",
                    name="포트폴리오 가치"
                ),
                row=1, col=1
            )
            
            # 수익률
            fig.add_trace(
                go.Scatter(
                    x=results["dates"],
                    y=results["returns"],
                    mode="lines",
                    name="수익률"
                ),
                row=2, col=1
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title="백테스팅 결과",
                xaxis_title="날짜",
                yaxis_title="가치 (USD)",
                yaxis2_title="수익률",
                showlegend=True
            )
            
            # 저장
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        except Exception as e:
            self.logger.error(f"결과 시각화 중 오류 발생: {e}")
            return go.Figure()
            
    def save_results(self,
                    results: Dict[str, Any],
                    save_path: str) -> None:
        """
        백테스팅 결과 저장
        
        Args:
            results: 백테스팅 결과
            save_path: 저장 경로
        """
        try:
            # 결과 저장
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
                
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {e}")
            raise