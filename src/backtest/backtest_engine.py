"""
백테스팅 엔진 모듈
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass
from ..strategy.integrated_strategy import IntegratedStrategy
from ..utils.database_manager import DatabaseManager
from ..strategy.base_strategy import BaseStrategy
from ..risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """백테스트 결과를 저장하는 데이터 클래스"""
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade: float
    avg_win: float
    avg_loss: float
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    monthly_returns: pd.DataFrame
    daily_returns: pd.DataFrame
    risk_metrics: Dict[str, float]

class BacktestEngine:
    """백테스팅 엔진 클래스"""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        risk_manager: Optional[RiskManager] = None,
        database_manager: Optional[DatabaseManager] = None
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.start_date = start_date
        self.end_date = end_date
        self.risk_manager = risk_manager
        self.database_manager = database_manager
        
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def run(self, data: pd.DataFrame) -> BacktestResult:
        """백테스트 실행"""
        try:
            self.logger.info("백테스트 시작")
            
            # 데이터 전처리
            data = self._preprocess_data(data)
            
            # 백테스트 실행
            for i in range(len(data)):
                current_data = data.iloc[i]
                timestamp = current_data.name
                
                # 포지션 관리
                await self._manage_positions(current_data, timestamp)
                
                # 전략 신호 생성
                signals = await self.strategy.generate_signals(current_data)
                
                # 거래 실행
                if signals:
                    await self._execute_trades(signals, current_data, timestamp)
                
                # 자본금 곡선 업데이트
                self._update_equity_curve(timestamp)
            
            # 결과 계산
            result = self._calculate_results()
            
            self.logger.info("백테스트 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리"""
        try:
            # 날짜 필터링
            if self.start_date:
                data = data[data.index >= self.start_date]
            if self.end_date:
                data = data[data.index <= self.end_date]
            
            # 결측치 처리
            data = data.fillna(method='ffill')
            
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            raise
    
    async def _manage_positions(self, current_data: pd.Series, timestamp: datetime):
        """포지션 관리"""
        try:
            for symbol, position in list(self.positions.items()):
                # 미실현 손익 계산
                current_price = current_data['close']
                entry_price = position['entry_price']
                size = position['size']
                
                if position['side'] == 'long':
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                
                # 손절/익절 체크
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                
                if (position['side'] == 'long' and current_price <= stop_loss) or \
                   (position['side'] == 'short' and current_price >= stop_loss):
                    await self._close_position(symbol, current_price, timestamp, 'stop_loss')
                elif (position['side'] == 'long' and current_price >= take_profit) or \
                     (position['side'] == 'short' and current_price <= take_profit):
                    await self._close_position(symbol, current_price, timestamp, 'take_profit')
            
        except Exception as e:
            self.logger.error(f"포지션 관리 중 오류 발생: {str(e)}")
            raise
    
    async def _execute_trades(self, signals: Dict[str, Any], current_data: pd.Series, timestamp: datetime):
        """거래 실행"""
        try:
            for symbol, signal in signals.items():
                if signal['action'] == 'buy' and symbol not in self.positions:
                    # 매수 신호
                    price = current_data['close'] * (1 + self.slippage)
                    size = self._calculate_position_size(price)
                    
                    if size > 0:
                        await self._open_position(symbol, 'long', price, size, timestamp)
                
                elif signal['action'] == 'sell' and symbol in self.positions:
                    # 매도 신호
                    price = current_data['close'] * (1 - self.slippage)
                    await self._close_position(symbol, price, timestamp, 'signal')
            
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류 발생: {str(e)}")
            raise
    
    def _calculate_position_size(self, price: float) -> float:
        """포지션 크기 계산"""
        try:
            if self.risk_manager:
                size = self.risk_manager.calculate_position_size(
                    price=price,
                    stop_loss=price * 0.95,  # 임시 손절가
                    confidence_level=0.95
                )
            else:
                size = (self.current_capital * 0.1) / price  # 기본값: 자본금의 10%
            
            return size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 중 오류 발생: {str(e)}")
            raise
    
    async def _open_position(self, symbol: str, side: str, price: float, size: float, timestamp: datetime):
        """포지션 오픈"""
        try:
            # 수수료 계산
            commission = price * size * self.commission
            
            # 포지션 생성
            self.positions[symbol] = {
                'side': side,
                'entry_price': price,
                'size': size,
                'entry_time': timestamp,
                'stop_loss': price * 0.95 if side == 'long' else price * 1.05,
                'take_profit': price * 1.1 if side == 'long' else price * 0.9
            }
            
            # 자본금 업데이트
            self.current_capital -= commission
            
            # 거래 기록
            trade = {
                'symbol': symbol,
                'side': side,
                'entry_price': price,
                'size': size,
                'entry_time': timestamp,
                'commission': commission
            }
            self.trades.append(trade)
            
            if self.database_manager:
                await self.database_manager.save_trade(trade)
            
        except Exception as e:
            self.logger.error(f"포지션 오픈 중 오류 발생: {str(e)}")
            raise
    
    async def _close_position(self, symbol: str, price: float, timestamp: datetime, reason: str):
        """포지션 클로즈"""
        try:
            position = self.positions[symbol]
            side = position['side']
            entry_price = position['entry_price']
            size = position['size']
            
            # 수수료 계산
            commission = price * size * self.commission
            
            # 손익 계산
            if side == 'long':
                pnl = (price - entry_price) * size - commission
            else:
                pnl = (entry_price - price) * size - commission
            
            # 자본금 업데이트
            self.current_capital += pnl
            
            # 포지션 제거
            del self.positions[symbol]
            
            # 거래 기록
            trade = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': price,
                'size': size,
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'pnl': pnl,
                'commission': commission,
                'reason': reason
            }
            self.trades.append(trade)
            
            if self.database_manager:
                await self.database_manager.save_trade(trade)
            
        except Exception as e:
            self.logger.error(f"포지션 클로즈 중 오류 발생: {str(e)}")
            raise
    
    def _update_equity_curve(self, timestamp: datetime):
        """자본금 곡선 업데이트"""
        try:
            # 미실현 손익 계산
            unrealized_pnl = 0
            for position in self.positions.values():
                if position['side'] == 'long':
                    unrealized_pnl += (position['current_price'] - position['entry_price']) * position['size']
                else:
                    unrealized_pnl += (position['entry_price'] - position['current_price']) * position['size']
            
            # 자본금 곡선 업데이트
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.current_capital + unrealized_pnl
            })
            
        except Exception as e:
            self.logger.error(f"자본금 곡선 업데이트 중 오류 발생: {str(e)}")
            raise
    
    def _calculate_results(self) -> BacktestResult:
        """백테스트 결과 계산"""
        try:
            # 거래 데이터프레임 생성
            trades_df = pd.DataFrame(self.trades)
            
            # 자본금 곡선 데이터프레임 생성
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # 수익률 계산
            returns = equity_df['equity'].pct_change().dropna()
            daily_returns = returns.resample('D').sum()
            monthly_returns = returns.resample('M').sum()
            
            # 성과 지표 계산
            total_return = (self.current_capital / self.initial_capital) - 1
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            max_drawdown = (equity_df['equity'].cummax() - equity_df['equity']) / equity_df['equity'].cummax()
            max_drawdown = max_drawdown.max()
            
            # 거래 통계
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 평균 거래
            avg_trade = trades_df['pnl'].mean()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            
            # 수익 요인
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # 리스크 지표
            risk_metrics = {
                'volatility': returns.std() * np.sqrt(252),
                'var_95': returns.quantile(0.05),
                'expected_shortfall': returns[returns <= returns.quantile(0.05)].mean(),
                'max_drawdown_duration': self._calculate_max_drawdown_duration(equity_df)
            }
            
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_capital=self.current_capital,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_trade=avg_trade,
                avg_win=avg_win,
                avg_loss=avg_loss,
                equity_curve=equity_df,
                trades=trades_df,
                monthly_returns=monthly_returns,
                daily_returns=daily_returns,
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            self.logger.error(f"백테스트 결과 계산 중 오류 발생: {str(e)}")
            raise
    
    def _calculate_max_drawdown_duration(self, equity_df: pd.DataFrame) -> int:
        """최대 낙폭 지속 기간 계산"""
        try:
            drawdown = (equity_df['equity'].cummax() - equity_df['equity']) / equity_df['equity'].cummax()
            drawdown_duration = 0
            max_duration = 0
            
            for d in drawdown:
                if d > 0:
                    drawdown_duration += 1
                    max_duration = max(max_duration, drawdown_duration)
                else:
                    drawdown_duration = 0
            
            return max_duration
            
        except Exception as e:
            self.logger.error(f"최대 낙폭 지속 기간 계산 중 오류 발생: {str(e)}")
            raise 