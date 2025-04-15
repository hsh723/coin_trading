import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy.optimize import minimize
from datetime import datetime, timedelta
import json
import os
from scipy.stats import norm

class PortfolioOptimizer:
    """포트폴리오 최적화 시스템"""
    
    def __init__(self,
                 config_path: str = "./config/portfolio_config.json",
                 data_dir: str = "./data",
                 log_dir: str = "./logs"):
        """
        포트폴리오 최적화 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
            data_dir: 데이터 디렉토리
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("portfolio_optimizer")
        
        # 포트폴리오 데이터
        self.returns = None
        self.covariance = None
        self.weights = None
        self.portfolio_value = None
        
        # 최적화 파라미터
        self.config = self._load_config()
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        self.max_leverage = self.config.get("max_leverage", 1.0)
        self.rebalance_threshold = self.config.get("rebalance_threshold", 0.05)
        self.max_position_size = self.config.get("max_position_size", 0.3)
        self.min_position_size = self.config.get("min_position_size", 0.05)
        
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
            
    def update_data(self, returns: pd.DataFrame) -> None:
        """
        포트폴리오 데이터 업데이트
        
        Args:
            returns: 자산별 수익률 데이터프레임
        """
        try:
            self.returns = returns
            self.covariance = returns.cov()
            self.logger.info("포트폴리오 데이터 업데이트 완료")
        except Exception as e:
            self.logger.error(f"데이터 업데이트 중 오류 발생: {e}")
            
    def calculate_efficient_frontier(self,
                                  returns: pd.DataFrame,
                                  num_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        효율적 프론티어 계산
        
        Args:
            returns: 수익률 데이터프레임
            num_points: 포인트 수
            
        Returns:
            수익률, 변동성 배열
        """
        try:
            # 평균 수익률과 공분산 행렬 계산
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # 최소 분산 포트폴리오 계산
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
            def portfolio_return(weights):
                return np.dot(weights.T, mean_returns)
                
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 가중치 합 = 1
            ]
            
            # 경계 조건
            bounds = [(0, 1) for _ in range(len(mean_returns))]
            
            # 목표 수익률 범위
            target_returns = np.linspace(
                mean_returns.min(),
                mean_returns.max(),
                num_points
            )
            
            # 효율적 프론티어 계산
            frontier_returns = []
            frontier_volatilities = []
            
            for target in target_returns:
                # 목표 수익률 제약 추가
                constraints.append(
                    {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target}
                )
                
                # 최적화
                result = minimize(
                    portfolio_volatility,
                    np.ones(len(mean_returns)) / len(mean_returns),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    frontier_returns.append(target)
                    frontier_volatilities.append(result.fun)
                    
                # 마지막 제약 제거
                constraints.pop()
                
            return np.array(frontier_returns), np.array(frontier_volatilities)
            
        except Exception as e:
            self.logger.error(f"효율적 프론티어 계산 중 오류 발생: {e}")
            return None, None
            
    def calculate_risk_parity_weights(self,
                                    returns: pd.DataFrame) -> np.ndarray:
        """
        리스크 패리티 가중치 계산
        
        Args:
            returns: 수익률 데이터프레임
            
        Returns:
            가중치 배열
        """
        try:
            # 공분산 행렬 계산
            cov_matrix = returns.cov()
            
            # 목적 함수 (리스크 기여도 차이의 제곱합)
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_vol
                target_contributions = np.ones_like(weights) / len(weights)
                return np.sum((risk_contributions - target_contributions) ** 2)
                
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 가중치 합 = 1
            ]
            
            # 경계 조건
            bounds = [(0, 1) for _ in range(len(returns.columns))]
            
            # 초기 가중치
            initial_weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            # 최적화
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x if result.success else None
            
        except Exception as e:
            self.logger.error(f"리스크 패리티 가중치 계산 중 오류 발생: {e}")
            return None
            
    def calculate_kelly_weights(self,
                              returns: pd.DataFrame) -> np.ndarray:
        """
        켈리 기준 가중치 계산
        
        Args:
            returns: 수익률 데이터프레임
            
        Returns:
            가중치 배열
        """
        try:
            # 평균 수익률과 변동성 계산
            mean_returns = returns.mean()
            volatilities = returns.std()
            
            # 켈리 가중치 계산
            kelly_weights = mean_returns / (volatilities ** 2)
            
            # 정규화
            kelly_weights = kelly_weights / np.sum(kelly_weights)
            
            # 포지션 크기 제한 적용
            kelly_weights = np.clip(
                kelly_weights,
                self.min_position_size,
                self.max_position_size
            )
            
            # 재정규화
            kelly_weights = kelly_weights / np.sum(kelly_weights)
            
            return kelly_weights
            
        except Exception as e:
            self.logger.error(f"켈리 기준 가중치 계산 중 오류 발생: {e}")
            return None
            
    def calculate_var(self,
                     returns: pd.DataFrame,
                     confidence_level: float = 0.95) -> float:
        """
        VaR (Value at Risk) 계산
        
        Args:
            returns: 수익률 데이터프레임
            confidence_level: 신뢰 수준
            
        Returns:
            VaR 값
        """
        try:
            portfolio_returns = returns.mean(axis=1)
            return np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        except Exception as e:
            self.logger.error(f"VaR 계산 중 오류 발생: {e}")
            return None
            
    def calculate_cvar(self,
                      returns: pd.DataFrame,
                      confidence_level: float = 0.95) -> float:
        """
        CVaR (Conditional Value at Risk) 계산
        
        Args:
            returns: 수익률 데이터프레임
            confidence_level: 신뢰 수준
            
        Returns:
            CVaR 값
        """
        try:
            portfolio_returns = returns.mean(axis=1)
            var = self.calculate_var(returns, confidence_level)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            return cvar
        except Exception as e:
            self.logger.error(f"CVaR 계산 중 오류 발생: {e}")
            return None
            
    def analyze_portfolio(self,
                         returns: pd.DataFrame,
                         weights: np.ndarray) -> Dict[str, Any]:
        """
        포트폴리오 분석
        
        Args:
            returns: 수익률 데이터프레임
            weights: 자산 가중치
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 포트폴리오 수익률 계산
            portfolio_returns = returns.dot(weights)
            
            # 기본 통계량
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.risk_free_rate) / volatility
            
            # 최대 낙폭
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # 리스크 기여도
            cov_matrix = returns.cov()
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_vol
            
            # 상관관계
            correlation_matrix = returns.corr()
            
            results = {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "risk_contributions": risk_contributions,
                "correlation_matrix": correlation_matrix
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"포트폴리오 분석 중 오류 발생: {e}")
            return {}
            
    def rebalance_portfolio(self,
                          current_weights: np.ndarray,
                          target_weights: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        포트폴리오 리밸런싱
        
        Args:
            current_weights: 현재 가중치
            target_weights: 목표 가중치
            
        Returns:
            새로운 가중치, 리밸런싱 필요 여부
        """
        try:
            # 가중치 차이 계산
            weight_diff = np.abs(current_weights - target_weights)
            
            # 리밸런싱 필요 여부 확인
            needs_rebalancing = np.any(weight_diff > self.rebalance_threshold)
            
            if needs_rebalancing:
                return target_weights, True
            else:
                return current_weights, False
                
        except Exception as e:
            self.logger.error(f"포트폴리오 리밸런싱 중 오류 발생: {e}")
            return current_weights, False
            
    def optimize_portfolio(self,
                         returns: pd.DataFrame,
                         target_return: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        포트폴리오 최적화
        
        Args:
            returns: 수익률 데이터프레임
            target_return: 목표 수익률 (선택)
            
        Returns:
            최적 가중치, 분석 결과
        """
        try:
            # 목적 함수 (포트폴리오 변동성)
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
                
            # 제약 조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 가중치 합 = 1
            ]
            
            if target_return is not None:
                constraints.append(
                    {'type': 'eq', 'fun': lambda x: np.dot(x, returns.mean()) - target_return}
                )
                
            # 경계 조건
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(len(returns.columns))]
            
            # 초기 가중치
            initial_weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            # 최적화
            result = minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                analysis_results = self.analyze_portfolio(returns, optimal_weights)
                return optimal_weights, analysis_results
            else:
                return None, {}
                
        except Exception as e:
            self.logger.error(f"포트폴리오 최적화 중 오류 발생: {e}")
            return None, {} 