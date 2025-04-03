"""
성과 분석 모듈

이 모듈은 트레이딩 시스템의 성과를 분석합니다.
주요 기능:
- 종합 성과 지표 계산
- 포지션별 분석
- 전략별 성과 비교
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    total_return: float  # 총 수익률
    daily_return: float  # 일일 수익률
    weekly_return: float  # 주간 수익률
    monthly_return: float  # 월간 수익률
    annual_return: float  # 연간 수익률
    win_rate: float  # 승률
    profit_factor: float  # 수익 팩터
    max_drawdown: float  # 최대 낙폭
    sharpe_ratio: float  # 샤프 비율
    sortino_ratio: float  # 소르티노 비율
    calmar_ratio: float  # 칼마 비율
    total_trades: int  # 총 거래 수
    winning_trades: int  # 수익 거래 수
    losing_trades: int  # 손실 거래 수
    average_return: float  # 평균 수익률
    return_std: float  # 수익률 표준편차
    average_win: float  # 평균 수익
    average_loss: float  # 평균 손실
    largest_win: float  # 최대 수익
    largest_loss: float  # 최대 손실
    average_hold_time: timedelta  # 평균 보유 시간

@dataclass
class PositionMetrics:
    symbol: str  # 심볼
    total_trades: int  # 총 거래 수
    winning_trades: int  # 수익 거래 수
    losing_trades: int  # 손실 거래 수
    win_rate: float  # 승률
    total_return: float  # 총 수익률
    average_return: float  # 평균 수익률
    profit_factor: float  # 수익 팩터
    average_hold_time: timedelta  # 평균 보유 시간
    max_drawdown: float  # 최대 낙폭

class PerformanceAnalyzer:
    """성과 분석 클래스"""
    
    def __init__(self, config: Dict):
        """
        성과 분석기 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config
        self.trades: List[Dict] = []
        self.positions: Dict[str, List[Dict]] = {}
        self.equity_curve: pd.Series = pd.Series()
        
        logger.info("PerformanceAnalyzer initialized")
    
    def add_trade(self, trade: Dict):
        """
        거래 추가
        
        Args:
            trade (Dict): 거래 정보
        """
        try:
            self.trades.append(trade)
            
            # 포지션별 거래 기록
            symbol = trade['symbol']
            if symbol not in self.positions:
                self.positions[symbol] = []
            self.positions[symbol].append(trade)
            
            # 자본금 곡선 업데이트
            self._update_equity_curve(trade)
            
        except Exception as e:
            logger.error(f"거래 추가 중 오류 발생: {str(e)}")
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """
        성과 지표 계산
        
        Returns:
            PerformanceMetrics: 성과 지표
        """
        try:
            if not self.trades:
                return None
            
            # 기본 지표 계산
            total_return = self._calculate_total_return()
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            max_drawdown = self._calculate_max_drawdown()
            
            # 수익률 지표 계산
            daily_return = self._calculate_period_return('daily')
            weekly_return = self._calculate_period_return('weekly')
            monthly_return = self._calculate_period_return('monthly')
            annual_return = self._calculate_period_return('annual')
            
            # 리스크 조정 수익률 지표 계산
            sharpe_ratio = self._calculate_sharpe_ratio()
            sortino_ratio = self._calculate_sortino_ratio()
            calmar_ratio = self._calculate_calmar_ratio()
            
            # 거래 통계 계산
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            average_return = self._calculate_average_return()
            return_std = self._calculate_return_std()
            
            # 수익/손실 통계 계산
            average_win = self._calculate_average_win()
            average_loss = self._calculate_average_loss()
            largest_win = self._calculate_largest_win()
            largest_loss = self._calculate_largest_loss()
            
            # 보유 시간 계산
            average_hold_time = self._calculate_average_hold_time()
            
            return PerformanceMetrics(
                total_return=total_return,
                daily_return=daily_return,
                weekly_return=weekly_return,
                monthly_return=monthly_return,
                annual_return=annual_return,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                average_return=average_return,
                return_std=return_std,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                average_hold_time=average_hold_time
            )
            
        except Exception as e:
            logger.error(f"성과 지표 계산 중 오류 발생: {str(e)}")
            raise
    
    def calculate_position_metrics(self, symbol: str) -> PositionMetrics:
        """
        포지션별 성과 지표 계산
        
        Args:
            symbol (str): 심볼
            
        Returns:
            PositionMetrics: 포지션 성과 지표
        """
        try:
            if symbol not in self.positions:
                return None
            
            trades = self.positions[symbol]
            
            # 기본 지표 계산
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 수익률 지표 계산
            total_return = sum(t['pnl'] for t in trades)
            average_return = total_return / total_trades if total_trades > 0 else 0
            profit_factor = self._calculate_profit_factor(trades)
            
            # 보유 시간 계산
            average_hold_time = self._calculate_average_hold_time(trades)
            
            # 최대 낙폭 계산
            max_drawdown = self._calculate_max_drawdown(trades)
            
            return PositionMetrics(
                symbol=symbol,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                average_return=average_return,
                profit_factor=profit_factor,
                average_hold_time=average_hold_time,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            logger.error(f"포지션 성과 지표 계산 중 오류 발생: {str(e)}")
            raise
    
    def _update_equity_curve(self, trade: Dict):
        """
        자본금 곡선 업데이트
        
        Args:
            trade (Dict): 거래 정보
        """
        try:
            timestamp = pd.Timestamp(trade['timestamp'])
            pnl = trade['pnl']
            
            if self.equity_curve.empty:
                self.equity_curve = pd.Series([pnl], index=[timestamp])
            else:
                self.equity_curve[timestamp] = self.equity_curve.iloc[-1] + pnl
            
        except Exception as e:
            logger.error(f"자본금 곡선 업데이트 중 오류 발생: {str(e)}")
    
    def _calculate_total_return(self) -> float:
        """
        총 수익률 계산
        
        Returns:
            float: 총 수익률
        """
        if not self.trades:
            return 0.0
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        return total_pnl / self.config['trading']['initial_capital']
    
    def _calculate_win_rate(self) -> float:
        """
        승률 계산
        
        Returns:
            float: 승률
        """
        if not self.trades:
            return 0.0
        
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        return winning_trades / len(self.trades)
    
    def _calculate_profit_factor(self, trades: Optional[List[Dict]] = None) -> float:
        """
        수익 팩터 계산
        
        Args:
            trades (Optional[List[Dict]]): 거래 목록
            
        Returns:
            float: 수익 팩터
        """
        if trades is None:
            trades = self.trades
            
        if not trades:
            return 0.0
        
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_max_drawdown(self, trades: Optional[List[Dict]] = None) -> float:
        """
        최대 낙폭 계산
        
        Args:
            trades (Optional[List[Dict]]): 거래 목록
            
        Returns:
            float: 최대 낙폭
        """
        if trades is None:
            trades = self.trades
            
        if not trades:
            return 0.0
        
        # 누적 수익률 계산
        cumulative_returns = np.cumsum([t['pnl'] for t in trades])
        
        # 최대 낙폭 계산
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (rolling_max - cumulative_returns) / rolling_max
        
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    def _calculate_period_return(self, period: str) -> float:
        """
        기간별 수익률 계산
        
        Args:
            period (str): 기간 (daily, weekly, monthly, annual)
            
        Returns:
            float: 수익률
        """
        if not self.trades:
            return 0.0
        
        # 거래 데이터를 DataFrame으로 변환
        df = pd.DataFrame(self.trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 기간별 수익률 계산
        if period == 'daily':
            returns = df.groupby(df['timestamp'].dt.date)['pnl'].sum()
        elif period == 'weekly':
            returns = df.groupby(df['timestamp'].dt.isocalendar().week)['pnl'].sum()
        elif period == 'monthly':
            returns = df.groupby(df['timestamp'].dt.month)['pnl'].sum()
        elif period == 'annual':
            returns = df.groupby(df['timestamp'].dt.year)['pnl'].sum()
        else:
            return 0.0
        
        return returns.mean() / self.config['trading']['initial_capital']
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        샤프 비율 계산
        
        Returns:
            float: 샤프 비율
        """
        if len(self.trades) < 2:
            return 0.0
        
        returns = pd.Series([t['pnl'] for t in self.trades]).pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        risk_free_rate = 0.02  # 연간 무위험 수익률 (2%)
        excess_returns = returns - risk_free_rate/252  # 일간 무위험 수익률
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    def _calculate_sortino_ratio(self) -> float:
        """
        소르티노 비율 계산
        
        Returns:
            float: 소르티노 비율
        """
        if len(self.trades) < 2:
            return 0.0
        
        returns = pd.Series([t['pnl'] for t in self.trades]).pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        risk_free_rate = 0.02  # 연간 무위험 수익률 (2%)
        excess_returns = returns - risk_free_rate/252  # 일간 무위험 수익률
        
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std() if downside_returns.std() > 0 else 0
    
    def _calculate_calmar_ratio(self) -> float:
        """
        칼마 비율 계산
        
        Returns:
            float: 칼마 비율
        """
        if not self.trades:
            return 0.0
        
        annual_return = self._calculate_period_return('annual')
        max_drawdown = self._calculate_max_drawdown()
        
        return annual_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    def _calculate_average_return(self) -> float:
        """
        평균 수익률 계산
        
        Returns:
            float: 평균 수익률
        """
        if not self.trades:
            return 0.0
        
        returns = [t['pnl'] / self.config['trading']['initial_capital'] for t in self.trades]
        return np.mean(returns)
    
    def _calculate_return_std(self) -> float:
        """
        수익률 표준편차 계산
        
        Returns:
            float: 수익률 표준편차
        """
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t['pnl'] / self.config['trading']['initial_capital'] for t in self.trades]
        return np.std(returns)
    
    def _calculate_average_win(self) -> float:
        """
        평균 수익 계산
        
        Returns:
            float: 평균 수익
        """
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        return np.mean(winning_trades) if winning_trades else 0.0
    
    def _calculate_average_loss(self) -> float:
        """
        평균 손실 계산
        
        Returns:
            float: 평균 손실
        """
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        return np.mean(losing_trades) if losing_trades else 0.0
    
    def _calculate_largest_win(self) -> float:
        """
        최대 수익 계산
        
        Returns:
            float: 최대 수익
        """
        winning_trades = [t['pnl'] for t in self.trades if t['pnl'] > 0]
        return max(winning_trades) if winning_trades else 0.0
    
    def _calculate_largest_loss(self) -> float:
        """
        최대 손실 계산
        
        Returns:
            float: 최대 손실
        """
        losing_trades = [t['pnl'] for t in self.trades if t['pnl'] < 0]
        return min(losing_trades) if losing_trades else 0.0
    
    def _calculate_average_hold_time(self, trades: Optional[List[Dict]] = None) -> timedelta:
        """
        평균 보유 시간 계산
        
        Args:
            trades (Optional[List[Dict]]): 거래 목록
            
        Returns:
            timedelta: 평균 보유 시간
        """
        if trades is None:
            trades = self.trades
            
        if not trades:
            return timedelta(0)
        
        hold_times = []
        for trade in trades:
            entry_time = pd.Timestamp(trade['entry_time'])
            exit_time = pd.Timestamp(trade['exit_time'])
            hold_times.append(exit_time - entry_time)
        
        return np.mean(hold_times) 