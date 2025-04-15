import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

# 로깅 설정
logger = logging.getLogger(__name__)

class TradeType(Enum):
    """거래 유형"""
    LONG = "LONG"    # 롱 포지션
    SHORT = "SHORT"  # 숏 포지션

@dataclass
class Trade:
    """거래 정보"""
    trade_id: str
    symbol: str
    trade_type: TradeType
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    commission: float
    
    @property
    def pnl(self) -> float:
        """손익 계산"""
        if self.trade_type == TradeType.LONG:
            return (self.exit_price - self.entry_price) * self.quantity - self.commission
        else:
            return (self.entry_price - self.exit_price) * self.quantity - self.commission
            
    @property
    def duration(self) -> float:
        """거래 기간 (초)"""
        return (self.exit_time - self.entry_time).total_seconds()
        
    @property
    def return_pct(self) -> float:
        """수익률 (%)"""
        return (self.pnl / (self.entry_price * self.quantity)) * 100

class BacktestResult:
    """백테스팅 결과"""
    def __init__(self,
                 trades: List[Trade],
                 portfolio_value: pd.Series,
                 initial_capital: float):
        """
        백테스팅 결과 초기화
        
        Args:
            trades: 거래 목록
            portfolio_value: 포트폴리오 가치 시계열
            initial_capital: 초기 자본금
        """
        self.trades = trades
        self.portfolio_value = portfolio_value
        self.initial_capital = initial_capital
        
    @property
    def total_return(self) -> float:
        """총 수익률"""
        return (self.portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
        
    @property
    def annual_return(self) -> float:
        """연간 수익률"""
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        return ((1 + self.total_return / 100) ** (365 / days) - 1) * 100
        
    @property
    def sharpe_ratio(self) -> float:
        """샤프 비율"""
        returns = self.portfolio_value.pct_change().dropna()
        return np.sqrt(252) * returns.mean() / returns.std()
        
    @property
    def max_drawdown(self) -> float:
        """최대 낙폭"""
        rolling_max = self.portfolio_value.expanding().max()
        drawdown = (self.portfolio_value - rolling_max) / rolling_max
        return drawdown.min() * 100
        
    @property
    def win_rate(self) -> float:
        """승률"""
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        return (winning_trades / len(self.trades)) * 100 if self.trades else 0
        
    @property
    def profit_factor(self) -> float:
        """손익비"""
        gross_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        결과 시각화
        
        Args:
            save_path: 저장 경로
        """
        plt.figure(figsize=(15, 10))
        
        # 포트폴리오 가치
        plt.subplot(2, 2, 1)
        plt.plot(self.portfolio_value)
        plt.title('포트폴리오 가치')
        plt.xlabel('시간')
        plt.ylabel('가치')
        
        # 일별 수익률
        plt.subplot(2, 2, 2)
        returns = self.portfolio_value.pct_change().dropna()
        sns.histplot(returns, kde=True)
        plt.title('일별 수익률 분포')
        plt.xlabel('수익률')
        plt.ylabel('빈도')
        
        # 거래 수익률
        plt.subplot(2, 2, 3)
        trade_returns = [trade.return_pct for trade in self.trades]
        plt.bar(range(len(trade_returns)), trade_returns)
        plt.title('거래별 수익률')
        plt.xlabel('거래 번호')
        plt.ylabel('수익률 (%)')
        
        # 누적 수익률
        plt.subplot(2, 2, 4)
        cumulative_returns = (1 + returns).cumprod() - 1
        plt.plot(cumulative_returns)
        plt.title('누적 수익률')
        plt.xlabel('시간')
        plt.ylabel('수익률')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class BacktestEngine:
    """백테스팅 엔진"""
    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        """
        백테스팅 엔진 초기화
        
        Args:
            data: OHLCV 데이터
            initial_capital: 초기 자본금
            commission: 수수료 비율
            slippage: 슬리피지 비율
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # 거래 기록
        self.trades = []
        self.current_position = 0
        self.current_price = 0
        self.portfolio_value = pd.Series(index=data.index)
        self.portfolio_value.iloc[0] = initial_capital
        
        # 로거 설정
        self.logger = logging.getLogger("backtest_engine")
        
    def run(self,
            strategy: Any,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None) -> BacktestResult:
        """
        백테스팅 실행
        
        Args:
            strategy: 트레이딩 전략
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            백테스팅 결과
        """
        try:
            # 데이터 필터링
            if start_date:
                self.data = self.data[self.data.index >= start_date]
            if end_date:
                self.data = self.data[self.data.index <= end_date]
                
            # 백테스팅 실행
            for i in range(1, len(self.data)):
                current_time = self.data.index[i]
                current_price = self.data.iloc[i]['close']
                self.current_price = current_price
                
                # 전략 신호 생성
                signal = strategy.generate_signal(self.data.iloc[:i+1])
                
                # 포지션 관리
                self._manage_position(signal, current_time)
                
                # 포트폴리오 가치 업데이트
                self._update_portfolio_value(current_time)
                
            # 최종 포지션 청산
            if self.current_position != 0:
                self._close_position(self.data.index[-1])
                
            return BacktestResult(
                trades=self.trades,
                portfolio_value=self.portfolio_value,
                initial_capital=self.initial_capital
            )
            
        except Exception as e:
            self.logger.error(f"백테스팅 실행 중 오류 발생: {e}")
            raise
            
    def _manage_position(self,
                        signal: int,
                        current_time: datetime) -> None:
        """
        포지션 관리
        
        Args:
            signal: 트레이딩 신호
            current_time: 현재 시간
        """
        try:
            # 롱 포지션 진입
            if signal == 1 and self.current_position <= 0:
                if self.current_position < 0:
                    self._close_position(current_time)
                self._open_position(TradeType.LONG, current_time)
                
            # 숏 포지션 진입
            elif signal == -1 and self.current_position >= 0:
                if self.current_position > 0:
                    self._close_position(current_time)
                self._open_position(TradeType.SHORT, current_time)
                
            # 포지션 청산
            elif signal == 0 and self.current_position != 0:
                self._close_position(current_time)
                
        except Exception as e:
            self.logger.error(f"포지션 관리 중 오류 발생: {e}")
            
    def _open_position(self,
                      trade_type: TradeType,
                      entry_time: datetime) -> None:
        """
        포지션 진입
        
        Args:
            trade_type: 거래 유형
            entry_time: 진입 시간
        """
        try:
            # 포지션 크기 계산
            position_size = self._calculate_position_size()
            
            # 거래 생성
            trade = Trade(
                trade_id=f"trade_{len(self.trades)}",
                symbol=self.data.name,
                trade_type=trade_type,
                entry_price=self.current_price * (1 + self.slippage if trade_type == TradeType.LONG else 1 - self.slippage),
                exit_price=0,
                quantity=position_size,
                entry_time=entry_time,
                exit_time=entry_time,
                commission=self.current_price * position_size * self.commission
            )
            
            # 포지션 업데이트
            self.current_position = position_size if trade_type == TradeType.LONG else -position_size
            self.trades.append(trade)
            
        except Exception as e:
            self.logger.error(f"포지션 진입 중 오류 발생: {e}")
            
    def _close_position(self, exit_time: datetime) -> None:
        """
        포지션 청산
        
        Args:
            exit_time: 청산 시간
        """
        try:
            if self.current_position == 0:
                return
                
            # 마지막 거래 업데이트
            last_trade = self.trades[-1]
            last_trade.exit_price = self.current_price * (1 - self.slippage if self.current_position > 0 else 1 + self.slippage)
            last_trade.exit_time = exit_time
            last_trade.commission += self.current_price * abs(self.current_position) * self.commission
            
            # 포지션 초기화
            self.current_position = 0
            
        except Exception as e:
            self.logger.error(f"포지션 청산 중 오류 발생: {e}")
            
    def _calculate_position_size(self) -> float:
        """
        포지션 크기 계산
        
        Returns:
            포지션 크기
        """
        # 포트폴리오 가치의 일정 비율로 포지션 크기 결정
        position_size_ratio = 0.1  # 포트폴리오의 10%
        return (self.portfolio_value.iloc[-1] * position_size_ratio) / self.current_price
        
    def _update_portfolio_value(self, current_time: datetime) -> None:
        """
        포트폴리오 가치 업데이트
        
        Args:
            current_time: 현재 시간
        """
        try:
            # 포지션 가치 계산
            position_value = self.current_position * self.current_price
            
            # 현금 가치 계산
            cash_value = self.portfolio_value.iloc[-1] - abs(self.current_position) * self.current_price
            
            # 포트폴리오 가치 업데이트
            self.portfolio_value[current_time] = position_value + cash_value
            
        except Exception as e:
            self.logger.error(f"포트폴리오 가치 업데이트 중 오류 발생: {e}")
            
    def save_results(self,
                    result: BacktestResult,
                    save_dir: str = "./results") -> None:
        """
        결과 저장
        
        Args:
            result: 백테스팅 결과
            save_dir: 저장 디렉토리
        """
        try:
            # 디렉토리 생성
            os.makedirs(save_dir, exist_ok=True)
            
            # 거래 기록 저장
            trades_df = pd.DataFrame([{
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'trade_type': trade.trade_type.value,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'duration': trade.duration,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time
            } for trade in result.trades])
            trades_df.to_csv(os.path.join(save_dir, "trades.csv"), index=False)
            
            # 포트폴리오 가치 저장
            result.portfolio_value.to_csv(os.path.join(save_dir, "portfolio_value.csv"))
            
            # 성과 지표 저장
            metrics = {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor
            }
            with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
                
            # 결과 시각화 저장
            result.plot_results(os.path.join(save_dir, "results.png"))
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {e}") 