"""
전략 포트폴리오 최적화 시스템
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from ..utils.database import DatabaseManager

class StrategyPortfolio:
    """전략 포트폴리오 클래스"""
    
    def __init__(self, db: DatabaseManager):
        """
        초기화
        
        Args:
            db (DatabaseManager): 데이터베이스 관리자
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.strategies = {}
        self.weights = {}
        
    async def add_strategy(self, strategy_id: str, strategy: Any) -> bool:
        """
        전략 추가
        
        Args:
            strategy_id (str): 전략 ID
            strategy (Any): 전략 객체
            
        Returns:
            bool: 추가 성공 여부
        """
        try:
            self.strategies[strategy_id] = strategy
            self.weights[strategy_id] = 1.0 / len(self.strategies)  # 균등 가중치로 초기화
            
            self.logger.info(f"전략 추가 완료: {strategy_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"전략 추가 실패: {str(e)}")
            return False
            
    async def optimize_weights(self, lookback_days: int = 30) -> Dict[str, float]:
        """
        전략 가중치 최적화
        
        Args:
            lookback_days (int): 과거 데이터 기간
            
        Returns:
            Dict[str, float]: 최적화된 가중치
        """
        try:
            # 전략별 수익률 데이터 수집
            returns_data = await self._collect_returns_data(lookback_days)
            
            if not returns_data:
                return self.weights
                
            # 수익률 데이터프레임 생성
            returns_df = pd.DataFrame(returns_data)
            
            # 공분산 행렬 계산
            cov_matrix = returns_df.cov()
            
            # 최적화 문제 설정
            n_strategies = len(self.strategies)
            initial_weights = np.ones(n_strategies) / n_strategies
            
            # 샤프 비율 최대화
            optimal_weights = self._maximize_sharpe_ratio(
                returns_df,
                cov_matrix,
                initial_weights
            )
            
            # 가중치 업데이트
            for i, strategy_id in enumerate(self.strategies.keys()):
                self.weights[strategy_id] = optimal_weights[i]
                
            # 최적화 결과 저장
            await self._save_optimization_result(optimal_weights)
            
            return self.weights
            
        except Exception as e:
            self.logger.error(f"가중치 최적화 실패: {str(e)}")
            return self.weights
            
    async def _collect_returns_data(self, lookback_days: int) -> Dict[str, List[float]]:
        """
        전략별 수익률 데이터 수집
        
        Args:
            lookback_days (int): 과거 데이터 기간
            
        Returns:
            Dict[str, List[float]]: 수익률 데이터
        """
        try:
            returns_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            for strategy_id in self.strategies:
                # 전략별 거래 기록 조회
                trades = await self.db.get_trades_by_strategy(
                    strategy_id,
                    start_date,
                    end_date
                )
                
                if trades:
                    # 일별 수익률 계산
                    daily_returns = self._calculate_daily_returns(trades)
                    returns_data[strategy_id] = daily_returns
                    
            return returns_data
            
        except Exception as e:
            self.logger.error(f"수익률 데이터 수집 실패: {str(e)}")
            return {}
            
    def _calculate_daily_returns(self, trades: List[Dict[str, Any]]) -> List[float]:
        """
        일별 수익률 계산
        
        Args:
            trades (List[Dict[str, Any]]): 거래 기록
            
        Returns:
            List[float]: 일별 수익률
        """
        try:
            # 거래 데이터프레임 생성
            trades_df = pd.DataFrame(trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df.set_index('timestamp', inplace=True)
            
            # 일별 수익률 계산
            daily_returns = trades_df['pnl'].resample('D').sum()
            return daily_returns.tolist()
            
        except Exception as e:
            self.logger.error(f"일별 수익률 계산 실패: {str(e)}")
            return []
            
    def _maximize_sharpe_ratio(
        self,
        returns_df: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        initial_weights: np.ndarray
    ) -> np.ndarray:
        """
        샤프 비율 최대화
        
        Args:
            returns_df (pd.DataFrame): 수익률 데이터프레임
            cov_matrix (pd.DataFrame): 공분산 행렬
            initial_weights (np.ndarray): 초기 가중치
            
        Returns:
            np.ndarray: 최적화된 가중치
        """
        try:
            # 목적 함수: 샤프 비율 최대화
            def objective(weights):
                portfolio_return = np.sum(returns_df.mean() * weights)
                portfolio_volatility = np.sqrt(
                    np.dot(weights.T, np.dot(cov_matrix, weights))
                )
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # 최소화 문제로 변환
                
            # 제약 조건: 가중치 합 = 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # 경계 조건: 0 <= 가중치 <= 1
            bounds = tuple((0, 1) for _ in range(len(initial_weights)))
            
            # 최적화 실행
            from scipy.optimize import minimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x
            
        except Exception as e:
            self.logger.error(f"샤프 비율 최대화 실패: {str(e)}")
            return initial_weights
            
    async def _save_optimization_result(self, weights: np.ndarray) -> bool:
        """
        최적화 결과 저장
        
        Args:
            weights (np.ndarray): 최적화된 가중치
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'weights': {
                    strategy_id: weight
                    for strategy_id, weight in zip(self.strategies.keys(), weights)
                }
            }
            
            await self.db.save_portfolio_optimization(result)
            return True
            
        except Exception as e:
            self.logger.error(f"최적화 결과 저장 실패: {str(e)}")
            return False
            
    async def get_portfolio_performance(self) -> Dict[str, Any]:
        """
        포트폴리오 성과 분석
        
        Returns:
            Dict[str, Any]: 성과 분석 결과
        """
        try:
            performance_data = {}
            
            for strategy_id, weight in self.weights.items():
                # 전략별 성과 데이터 조회
                strategy_performance = await self.db.get_strategy_performance(strategy_id)
                
                if strategy_performance:
                    # 가중치 적용
                    weighted_performance = {
                        'return': strategy_performance['return'] * weight,
                        'sharpe_ratio': strategy_performance['sharpe_ratio'],
                        'max_drawdown': strategy_performance['max_drawdown'],
                        'win_rate': strategy_performance['win_rate']
                    }
                    performance_data[strategy_id] = weighted_performance
                    
            # 포트폴리오 전체 성과 계산
            total_return = sum(
                perf['return']
                for perf in performance_data.values()
            )
            
            return {
                'strategies': performance_data,
                'total_return': total_return,
                'weights': self.weights
            }
            
        except Exception as e:
            self.logger.error(f"포트폴리오 성과 분석 실패: {str(e)}")
            return {} 