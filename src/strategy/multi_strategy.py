"""
멀티 전략 관리 모듈
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from .integrated_strategy import IntegratedStrategy

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """전략 성과 지표 데이터 클래스"""
    
    # 기본 지표
    strategy_name: str
    capital: float
    returns: float
    sharpe_ratio: float
    max_drawdown: float
    
    # 거래 지표
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade: float
    
    # 리스크 지표
    volatility: float
    var_95: float
    expected_shortfall: float
    
    # 최근 성과
    daily_returns: List[float]
    weekly_returns: List[float]
    monthly_returns: List[float]

class MultiStrategyManager:
    """멀티 전략 관리 클래스"""
    
    def __init__(
        self,
        initial_capital: float,
        max_strategies: int = 5,
        rebalance_threshold: float = 0.1,
        min_capital_per_strategy: float = 0.1
    ):
        """
        초기화
        
        Args:
            initial_capital: 초기 자본금
            max_strategies: 최대 전략 수
            rebalance_threshold: 리밸런싱 임계값
            min_capital_per_strategy: 전략당 최소 자본금 비율
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_strategies = max_strategies
        self.rebalance_threshold = rebalance_threshold
        self.min_capital_per_strategy = min_capital_per_strategy
        
        # 전략 관리
        self.strategies: Dict[str, Dict] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.strategy_weights: Dict[str, float] = {}
        
        # 성과 기록
        self.daily_returns: List[float] = []
        self.weekly_returns: List[float] = []
        self.monthly_returns: List[float] = []
    
    def add_strategy(
        self,
        name: str,
        strategy: IntegratedStrategy,
        initial_weight: float = 0.2
    ) -> bool:
        """
        전략 추가
        
        Args:
            name: 전략 이름
            strategy: 전략 객체
            initial_weight: 초기 가중치
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 최대 전략 수 체크
            if len(self.strategies) >= self.max_strategies:
                logger.warning(f"최대 전략 수 초과: {self.max_strategies}")
                return False
            
            # 가중치 합 체크
            total_weight = sum(self.strategy_weights.values()) + initial_weight
            if total_weight > 1.0:
                logger.warning("가중치 합이 1을 초과합니다.")
                return False
            
            # 전략 추가
            self.strategies[name] = {
                'strategy': strategy,
                'capital': self.current_capital * initial_weight,
                'trades': [],
                'positions': []
            }
            
            # 가중치 설정
            self.strategy_weights[name] = initial_weight
            
            # 성과 지표 초기화
            self.strategy_metrics[name] = StrategyMetrics(
                strategy_name=name,
                capital=self.current_capital * initial_weight,
                returns=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade=0.0,
                volatility=0.0,
                var_95=0.0,
                expected_shortfall=0.0,
                daily_returns=[],
                weekly_returns=[],
                monthly_returns=[]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"전략 추가 중 오류 발생: {str(e)}")
            return False
    
    def remove_strategy(self, name: str) -> bool:
        """
        전략 제거
        
        Args:
            name: 전략 이름
            
        Returns:
            bool: 성공 여부
        """
        try:
            if name not in self.strategies:
                logger.warning(f"존재하지 않는 전략: {name}")
                return False
            
            # 전략 제거
            del self.strategies[name]
            del self.strategy_metrics[name]
            
            # 가중치 재분배
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for strategy_name in self.strategy_weights:
                    self.strategy_weights[strategy_name] /= total_weight
            
            return True
            
        except Exception as e:
            logger.error(f"전략 제거 중 오류 발생: {str(e)}")
            return False
    
    def update_strategy_metrics(self, name: str, market_data: pd.DataFrame) -> bool:
        """
        전략 성과 지표 업데이트
        
        Args:
            name: 전략 이름
            market_data: 시장 데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            if name not in self.strategies:
                return False
            
            strategy = self.strategies[name]
            metrics = self.strategy_metrics[name]
            
            # 거래 내역 업데이트
            trades = strategy['trades']
            if trades:
                # 승률 계산
                winning_trades = [t for t in trades if t['pnl'] > 0]
                metrics.win_rate = len(winning_trades) / len(trades)
                
                # 손익비 계산
                total_profit = sum(t['pnl'] for t in winning_trades)
                total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
                metrics.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # 평균 거래 수익 계산
                metrics.avg_trade = sum(t['pnl'] for t in trades) / len(trades)
            
            # 수익률 계산
            if strategy['capital'] > 0:
                metrics.returns = (strategy['capital'] - self.initial_capital * self.strategy_weights[name]) / (self.initial_capital * self.strategy_weights[name])
            
            # 변동성 계산
            if metrics.daily_returns:
                returns = pd.Series(metrics.daily_returns)
                metrics.volatility = returns.std() * np.sqrt(252)  # 연간화
                metrics.var_95 = np.percentile(returns, 5)
                metrics.expected_shortfall = returns[returns <= metrics.var_95].mean()
            
            # 샤프 비율 계산
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.returns / metrics.volatility
            
            # 최대 낙폭 계산
            if metrics.daily_returns:
                cumulative_returns = np.cumprod(1 + np.array(metrics.daily_returns))
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (peak - cumulative_returns) / peak
                metrics.max_drawdown = np.max(drawdown)
            
            return True
            
        except Exception as e:
            logger.error(f"전략 성과 지표 업데이트 중 오류 발생: {str(e)}")
            return False
    
    def should_rebalance(self) -> Tuple[bool, Dict[str, float]]:
        """
        리밸런싱 필요 여부 확인
        
        Returns:
            Tuple[bool, Dict[str, float]]: (리밸런싱 필요 여부, 새로운 가중치)
        """
        try:
            new_weights = {}
            needs_rebalance = False
            
            # 현재 자본금 계산
            total_capital = sum(strategy['capital'] for strategy in self.strategies.values())
            
            # 가중치 계산
            for name, strategy in self.strategies.items():
                current_weight = strategy['capital'] / total_capital
                target_weight = self.strategy_weights[name]
                
                # 가중치 차이 확인
                if abs(current_weight - target_weight) > self.rebalance_threshold:
                    needs_rebalance = True
                
                new_weights[name] = target_weight
            
            return needs_rebalance, new_weights
            
        except Exception as e:
            logger.error(f"리밸런싱 필요 여부 확인 중 오류 발생: {str(e)}")
            return False, {}
    
    def rebalance(self, new_weights: Dict[str, float]) -> bool:
        """
        포트폴리오 리밸런싱
        
        Args:
            new_weights: 새로운 가중치
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 현재 자본금 계산
            total_capital = sum(strategy['capital'] for strategy in self.strategies.values())
            
            # 자본금 재분배
            for name, weight in new_weights.items():
                target_capital = total_capital * weight
                current_capital = self.strategies[name]['capital']
                
                # 자본금 조정
                if target_capital > current_capital:
                    # 자본금 추가
                    self.strategies[name]['capital'] = target_capital
                else:
                    # 자본금 감소
                    self.strategies[name]['capital'] = target_capital
            
            return True
            
        except Exception as e:
            logger.error(f"리밸런싱 중 오류 발생: {str(e)}")
            return False
    
    def get_strategy_signals(self, market_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        전략별 매매 신호 조회
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            Dict[str, Dict]: 전략별 매매 신호
        """
        signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                # 매매 신호 생성
                signal = strategy['strategy'].generate_signals(market_data)
                if signal:
                    signals[name] = signal
            except Exception as e:
                logger.error(f"전략 {name} 매매 신호 생성 중 오류 발생: {str(e)}")
        
        return signals
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        포트폴리오 성과 지표 조회
        
        Returns:
            Dict[str, Any]: 포트폴리오 성과 지표
        """
        try:
            # 포트폴리오 자본금 계산
            total_capital = sum(strategy['capital'] for strategy in self.strategies.values())
            
            # 포트폴리오 수익률 계산
            total_returns = (total_capital - self.initial_capital) / self.initial_capital
            
            # 포트폴리오 변동성 계산
            if self.daily_returns:
                returns = pd.Series(self.daily_returns)
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = total_returns / volatility if volatility > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
            
            # 최대 낙폭 계산
            if self.daily_returns:
                cumulative_returns = np.cumprod(1 + np.array(self.daily_returns))
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = (peak - cumulative_returns) / peak
                max_drawdown = np.max(drawdown)
            else:
                max_drawdown = 0
            
            return {
                'total_capital': total_capital,
                'total_returns': total_returns,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'strategy_weights': self.strategy_weights,
                'strategy_metrics': {
                    name: {
                        'returns': metrics.returns,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown': metrics.max_drawdown,
                        'win_rate': metrics.win_rate
                    }
                    for name, metrics in self.strategy_metrics.items()
                }
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 성과 지표 계산 중 오류 발생: {str(e)}")
            return {} 