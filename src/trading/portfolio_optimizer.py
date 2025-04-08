"""
포트폴리오 최적화 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from src.utils.logger import setup_logger

logger = setup_logger()

class PortfolioOptimizer:
    def __init__(
        self,
        config: Dict[str, Any],
        initial_capital: float
    ):
        """
        포트폴리오 최적화기 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
            initial_capital (float): 초기 자본금
        """
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.logger = setup_logger()
        
        # 포트폴리오 설정
        self.symbols = config.get('portfolio', {}).get('symbols', [])
        self.timeframe = config.get('portfolio', {}).get('timeframe', '1d')
        self.rebalance_period = config.get('portfolio', {}).get('rebalance_period', 7)  # 일
        self.min_weight = config.get('portfolio', {}).get('min_weight', 0.1)  # 최소 비중
        self.max_weight = config.get('portfolio', {}).get('max_weight', 0.4)  # 최대 비중
        
        # 상태 변수
        self.weights: Dict[str, float] = {}
        self.last_rebalance_date: Optional[datetime] = None
        self.positions: Dict[str, Dict[str, Any]] = {}
        
    def calculate_returns(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        수익률 계산
        
        Args:
            market_data (Dict[str, pd.DataFrame]): 시장 데이터
            
        Returns:
            pd.DataFrame: 수익률 데이터
        """
        try:
            returns = pd.DataFrame()
            
            for symbol, data in market_data.items():
                if symbol in self.symbols:
                    # 종가 기준 수익률 계산
                    returns[symbol] = data['close'].pct_change()
                    
            return returns.dropna()
            
        except Exception as e:
            self.logger.error(f"수익률 계산 실패: {str(e)}")
            return pd.DataFrame()
            
    def calculate_covariance(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        공분산 행렬 계산
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            
        Returns:
            pd.DataFrame: 공분산 행렬
        """
        try:
            return returns.cov()
            
        except Exception as e:
            self.logger.error(f"공분산 행렬 계산 실패: {str(e)}")
            return pd.DataFrame()
            
    def optimize_weights(
        self,
        returns: pd.DataFrame,
        covariance: pd.DataFrame
    ) -> Dict[str, float]:
        """
        최적 가중치 계산
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            covariance (pd.DataFrame): 공분산 행렬
            
        Returns:
            Dict[str, float]: 최적 가중치
        """
        try:
            # 샤프 비율 최대화를 위한 최적화
            from scipy.optimize import minimize
            
            def negative_sharpe(weights):
                portfolio_return = np.sum(returns.mean() * weights)
                portfolio_volatility = np.sqrt(
                    np.dot(weights.T, np.dot(covariance, weights))
                )
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio
                
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 가중치 합 = 1
                {'type': 'ineq', 'fun': lambda x: x - self.min_weight},  # 최소 비중
                {'type': 'ineq', 'fun': lambda x: self.max_weight - x}  # 최대 비중
            ]
            
            # 초기 가중치
            initial_weights = np.array([1/len(self.symbols)] * len(self.symbols))
            
            # 최적화 실행
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1)] * len(self.symbols)
            )
            
            # 결과를 딕셔너리로 변환
            weights = dict(zip(self.symbols, result.x))
            
            return weights
            
        except Exception as e:
            self.logger.error(f"최적 가중치 계산 실패: {str(e)}")
            return {}
            
    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        covariance: pd.DataFrame
    ) -> Dict[str, float]:
        """
        포트폴리오 지표 계산
        
        Args:
            weights (Dict[str, float]): 가중치
            returns (pd.DataFrame): 수익률 데이터
            covariance (pd.DataFrame): 공분산 행렬
            
        Returns:
            Dict[str, float]: 포트폴리오 지표
        """
        try:
            # 포트폴리오 수익률
            portfolio_return = np.sum(returns.mean() * list(weights.values()))
            
            # 포트폴리오 변동성
            portfolio_volatility = np.sqrt(
                np.dot(list(weights.values()), np.dot(covariance, list(weights.values())))
            )
            
            # 샤프 비율
            risk_free_rate = 0.02 / 252  # 연간 2% 가정
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            # 최대 낙폭
            portfolio_returns = np.sum(returns * list(weights.values()), axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())
            
            return {
                'return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"포트폴리오 지표 계산 실패: {str(e)}")
            return {}
            
    def check_rebalance_needed(self) -> bool:
        """
        리밸런싱 필요 여부 확인
        
        Returns:
            bool: 리밸런싱 필요 여부
        """
        try:
            if not self.last_rebalance_date:
                return True
                
            # 마지막 리밸런싱 이후 경과일 확인
            days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
            
            return days_since_rebalance >= self.rebalance_period
            
        except Exception as e:
            self.logger.error(f"리밸런싱 필요 여부 확인 실패: {str(e)}")
            return False
            
    def update_portfolio(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        포트폴리오 업데이트
        
        Args:
            market_data (Dict[str, pd.DataFrame]): 시장 데이터
            current_positions (Dict[str, Dict[str, Any]]): 현재 포지션
            
        Returns:
            Dict[str, Any]: 포트폴리오 업데이트 결과
        """
        try:
            # 리밸런싱 필요 여부 확인
            if not self.check_rebalance_needed():
                return {
                    'rebalance_needed': False,
                    'weights': self.weights,
                    'metrics': self.calculate_portfolio_metrics(
                        self.weights,
                        self.calculate_returns(market_data),
                        self.calculate_covariance(self.calculate_returns(market_data))
                    )
                }
                
            # 수익률 계산
            returns = self.calculate_returns(market_data)
            if returns.empty:
                return {'rebalance_needed': False, 'error': '수익률 계산 실패'}
                
            # 공분산 행렬 계산
            covariance = self.calculate_covariance(returns)
            if covariance.empty:
                return {'rebalance_needed': False, 'error': '공분산 행렬 계산 실패'}
                
            # 최적 가중치 계산
            weights = self.optimize_weights(returns, covariance)
            if not weights:
                return {'rebalance_needed': False, 'error': '최적 가중치 계산 실패'}
                
            # 포트폴리오 지표 계산
            metrics = self.calculate_portfolio_metrics(weights, returns, covariance)
            
            # 포지션 조정 필요 여부 확인
            position_adjustments = {}
            for symbol in self.symbols:
                target_value = self.current_capital * weights[symbol]
                current_position = current_positions.get(symbol, {})
                current_value = current_position.get('value', 0)
                
                if abs(target_value - current_value) > self.current_capital * 0.01:  # 1% 이상 차이
                    position_adjustments[symbol] = {
                        'target_value': target_value,
                        'current_value': current_value,
                        'adjustment': target_value - current_value
                    }
                    
            # 결과 반환
            result = {
                'rebalance_needed': True,
                'weights': weights,
                'metrics': metrics,
                'position_adjustments': position_adjustments
            }
            
            # 리밸런싱 날짜 업데이트
            self.last_rebalance_date = datetime.now()
            
            # 가중치 업데이트
            self.weights = weights
            
            return result
            
        except Exception as e:
            self.logger.error(f"포트폴리오 업데이트 실패: {str(e)}")
            return {'rebalance_needed': False, 'error': str(e)}
            
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        포트폴리오 상태 조회
        
        Returns:
            Dict[str, Any]: 포트폴리오 상태
        """
        try:
            return {
                'current_capital': self.current_capital,
                'weights': self.weights,
                'positions': self.positions,
                'last_rebalance_date': self.last_rebalance_date,
                'symbols': self.symbols
            }
            
        except Exception as e:
            self.logger.error(f"포트폴리오 상태 조회 실패: {str(e)}")
            return {} 