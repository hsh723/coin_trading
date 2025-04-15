import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union, Any

class PortfolioOptimizer:
    """포트폴리오 최적화 클래스
    
    다양한 최적화 방법을 제공:
    - 효율적 프론티어 계산
    - 샤프 비율 최적화
    - 리스크 패리티 (위험균형) 포트폴리오
    - 최소 분산 포트폴리오
    - 최대 다각화 포트폴리오
    """
    def __init__(self, risk_free_rate: float = 0.02, target_return: Optional[float] = None):
        """
        Args:
            risk_free_rate: 무위험 수익률
            target_return: 목표 수익률 (None이면 샤프 비율 최적화)
        """
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.optimal_weights = None
        self.efficient_frontier = None
        
    def optimize(self, returns: pd.DataFrame, 
                 method: str = 'sharpe', 
                 constraints: Optional[List[Dict]] = None) -> Dict[str, float]:
        """포트폴리오 최적화 수행
        
        Args:
            returns: 자산 수익률 데이터프레임 (열은 자산, 행은 기간)
            method: 최적화 방법 ('sharpe', 'min_var', 'risk_parity', 'max_div', 'target_return')
            constraints: 추가 제약 조건 리스트
            
        Returns:
            최적 자산 배분 비율 딕셔너리
        """
        if method == 'sharpe':
            return self._optimize_sharpe(returns, constraints)
        elif method == 'min_var':
            return self._optimize_min_variance(returns, constraints)
        elif method == 'risk_parity':
            return self._optimize_risk_parity(returns, constraints)
        elif method == 'max_div':
            return self._optimize_max_diversification(returns, constraints)
        elif method == 'target_return':
            if self.target_return is None:
                raise ValueError("target_return 파라미터가 설정되지 않았습니다")
            return self._optimize_target_return(returns, self.target_return, constraints)
        else:
            raise ValueError(f"지원하지 않는 최적화 방법: {method}")
    
    def _optimize_sharpe(self, returns: pd.DataFrame, 
                        constraints: Optional[List[Dict]] = None) -> Dict[str, float]:
        """샤프 비율 최적화
        
        Args:
            returns: 자산 수익률 데이터프레임
            constraints: 추가 제약 조건
            
        Returns:
            최적 가중치 딕셔너리
        """
        n_assets = len(returns.columns)
        
        # 기본 제약 조건: 가중치 합 = 1, 가중치 >= 0
        base_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # 추가 제약 조건
        if constraints is not None:
            base_constraints.extend(constraints)
        
        # 목적 함수: 샤프 비율 최대화
        def objective(weights):
            weights = np.array(weights)
            port_return = np.sum(returns.mean() * weights) * 252
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility
            return -sharpe_ratio  # 최소화 문제이므로 음수 취함
        
        # 초기 가중치: 균등 배분
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 최적화 실행
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=base_constraints)
        
        if not result['success']:
            raise ValueError(f"최적화 실패: {result['message']}")
        
        # 결과 가중치
        self.optimal_weights = dict(zip(returns.columns, result['x']))
        return self.optimal_weights
    
    def _optimize_min_variance(self, returns: pd.DataFrame, 
                              constraints: Optional[List[Dict]] = None) -> Dict[str, float]:
        """최소 분산 포트폴리오 최적화
        
        Args:
            returns: 자산 수익률 데이터프레임
            constraints: 추가 제약 조건
            
        Returns:
            최적 가중치 딕셔너리
        """
        n_assets = len(returns.columns)
        
        # 기본 제약 조건: 가중치 합 = 1, 가중치 >= 0
        base_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # 추가 제약 조건
        if constraints is not None:
            base_constraints.extend(constraints)
        
        # 목적 함수: 포트폴리오 분산 최소화
        def objective(weights):
            weights = np.array(weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            return port_volatility
        
        # 초기 가중치: 균등 배분
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 최적화 실행
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=base_constraints)
        
        if not result['success']:
            raise ValueError(f"최적화 실패: {result['message']}")
        
        # 결과 가중치
        self.optimal_weights = dict(zip(returns.columns, result['x']))
        return self.optimal_weights
    
    def _optimize_risk_parity(self, returns: pd.DataFrame, 
                             constraints: Optional[List[Dict]] = None) -> Dict[str, float]:
        """리스크 패리티 포트폴리오 최적화
        
        각 자산이 포트폴리오 위험에 동일하게 기여하도록 가중치 설정
        
        Args:
            returns: 자산 수익률 데이터프레임
            constraints: 추가 제약 조건
            
        Returns:
            최적 가중치 딕셔너리
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values * 252
        
        # 목적 함수: 위험 기여도 편차 최소화
        def objective(weights):
            weights = np.array(weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_contribution = weights * (np.dot(cov_matrix, weights)) / port_volatility
            target_risk_contribution = port_volatility / n_assets
            risk_diff = np.sum((risk_contribution - target_risk_contribution)**2)
            return risk_diff
        
        # 기본 제약 조건: 가중치 합 = 1, 가중치 >= 0
        base_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # 추가 제약 조건
        if constraints is not None:
            base_constraints.extend(constraints)
        
        # 초기 가중치: 균등 배분
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 최적화 실행
        bounds = tuple((0.01, 1) for _ in range(n_assets))  # 최소 가중치 0.01 설정
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=base_constraints)
        
        if not result['success']:
            raise ValueError(f"최적화 실패: {result['message']}")
        
        # 결과 가중치
        self.optimal_weights = dict(zip(returns.columns, result['x']))
        return self.optimal_weights
    
    def _optimize_max_diversification(self, returns: pd.DataFrame, 
                                    constraints: Optional[List[Dict]] = None) -> Dict[str, float]:
        """최대 다각화 포트폴리오 최적화
        
        다각화 비율(DR)을 최대화하는 포트폴리오
        DR = (가중 평균 변동성) / (포트폴리오 변동성)
        
        Args:
            returns: 자산 수익률 데이터프레임
            constraints: 추가 제약 조건
            
        Returns:
            최적 가중치 딕셔너리
        """
        n_assets = len(returns.columns)
        volatilities = np.sqrt(np.diag(returns.cov().values * 252))
        cov_matrix = returns.cov().values * 252
        
        # 목적 함수: 다각화 비율의 역수 최소화
        def objective(weights):
            weights = np.array(weights)
            weighted_avg_vol = np.sum(weights * volatilities)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            diversification_ratio = weighted_avg_vol / port_vol
            return -diversification_ratio  # 최소화 문제이므로 음수 취함
        
        # 기본 제약 조건: 가중치 합 = 1, 가중치 >= 0
        base_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # 추가 제약 조건
        if constraints is not None:
            base_constraints.extend(constraints)
        
        # 초기 가중치: 균등 배분
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 최적화 실행
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=base_constraints)
        
        if not result['success']:
            raise ValueError(f"최적화 실패: {result['message']}")
        
        # 결과 가중치
        self.optimal_weights = dict(zip(returns.columns, result['x']))
        return self.optimal_weights
    
    def _optimize_target_return(self, returns: pd.DataFrame, 
                              target_return: float,
                              constraints: Optional[List[Dict]] = None) -> Dict[str, float]:
        """목표 수익률을 달성하면서 분산을 최소화하는 포트폴리오
        
        Args:
            returns: 자산 수익률 데이터프레임
            target_return: 목표 수익률 (연율화 기준)
            constraints: 추가 제약 조건
            
        Returns:
            최적 가중치 딕셔너리
        """
        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        
        # 기본 제약 조건: 가중치 합 = 1, 가중치 >= 0, 수익률 = target_return
        base_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(w * mean_returns) - target_return}
        ]
        
        # 추가 제약 조건
        if constraints is not None:
            base_constraints.extend(constraints)
        
        # 목적 함수: 포트폴리오 분산 최소화
        def objective(weights):
            weights = np.array(weights)
            port_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
            return port_variance
        
        # 초기 가중치: 균등 배분
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 최적화 실행
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=base_constraints)
        
        if not result['success']:
            raise ValueError(f"최적화 실패: {result['message']}")
        
        # 결과 가중치
        self.optimal_weights = dict(zip(returns.columns, result['x']))
        return self.optimal_weights
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame, n_points: int = 50,
                                    constraints: Optional[List[Dict]] = None) -> pd.DataFrame:
        """효율적 프론티어 계산
        
        Args:
            returns: 자산 수익률 데이터프레임
            n_points: 프론티어 상의 포인트 수
            constraints: 추가 제약 조건
            
        Returns:
            효율적 프론티어 데이터프레임 (수익률, 위험)
        """
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # 최소 및 최대 수익률 범위 계산
        min_return = self._optimize_min_variance(returns, constraints)
        min_ret = np.sum(list(min_return.values()) * mean_returns)
        
        # 최대 수익률: 가장 높은 수익률 자산에 모두 투자
        max_idx = mean_returns.idxmax()
        max_ret = mean_returns[max_idx]
        
        # 수익률 범위
        target_returns = np.linspace(min_ret, max_ret, n_points)
        efficient_frontier = []
        
        for target_ret in target_returns:
            self.target_return = target_ret
            weights = self._optimize_target_return(returns, target_ret, constraints)
            
            # 포트폴리오 통계
            port_std = np.sqrt(np.dot(list(weights.values()), np.dot(cov_matrix, list(weights.values()))))
            sharpe = (target_ret - self.risk_free_rate) / port_std
            
            efficient_frontier.append({
                'Return': target_ret,
                'Risk': port_std,
                'Sharpe': sharpe,
                'Weights': weights
            })
        
        self.efficient_frontier = pd.DataFrame(efficient_frontier)
        return self.efficient_frontier
    
    def plot_efficient_frontier(self, show_assets: bool = True, 
                              highlight_optimal: bool = True,
                              title: str = '효율적 프론티어') -> None:
        """효율적 프론티어 시각화
        
        Args:
            show_assets: 개별 자산 표시 여부
            highlight_optimal: 최적 포트폴리오 강조 표시 여부
            title: 그래프 제목
        """
        if self.efficient_frontier is None:
            raise ValueError("먼저 calculate_efficient_frontier를 호출하세요")
        
        plt.figure(figsize=(12, 6))
        
        # 효율적 프론티어 플롯
        plt.plot(self.efficient_frontier['Risk'], 
                self.efficient_frontier['Return'], 
                'b-', linewidth=2, label='효율적 프론티어')
        
        # 최적 포트폴리오 강조
        if highlight_optimal and self.optimal_weights is not None:
            optimal_idx = self.efficient_frontier['Sharpe'].idxmax()
            opt_risk = self.efficient_frontier.loc[optimal_idx, 'Risk']
            opt_return = self.efficient_frontier.loc[optimal_idx, 'Return']
            
            plt.scatter(opt_risk, opt_return, s=100, c='red', 
                       marker='*', label='최적 포트폴리오')
        
        # 개별 자산 플롯
        if show_assets and self.optimal_weights is not None:
            assets = list(self.optimal_weights.keys())
            returns_df = pd.DataFrame(self.efficient_frontier.iloc[0]['Weights']).T
            returns_df.columns = assets
            
            # 연간 수익률 및 표준편차 계산
            annual_returns = returns_df.mean() * 252
            annual_std = returns_df.std() * np.sqrt(252)
            
            plt.scatter(annual_std, annual_returns, s=50, c='green', 
                       alpha=0.7, label='개별 자산')
            
            # 자산 이름 레이블 추가
            for i, asset in enumerate(assets):
                plt.annotate(asset, (annual_std[i], annual_returns[i]), 
                            xytext=(5, 5), textcoords='offset points')
        
        plt.title(title)
        plt.xlabel('위험 (연간 표준편차)')
        plt.ylabel('기대 수익률 (연간)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_portfolio_stats(self, weights: Dict[str, float], returns: pd.DataFrame) -> Dict[str, float]:
        """포트폴리오 통계 계산
        
        Args:
            weights: 자산 가중치 딕셔너리
            returns: 자산 수익률 데이터프레임
            
        Returns:
            포트폴리오 통계 딕셔너리
        """
        # 가중치 리스트로 변환
        weights_list = [weights[asset] for asset in returns.columns]
        
        # 포트폴리오 통계 계산
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        port_return = np.sum(weights_list * mean_returns)
        port_std = np.sqrt(np.dot(weights_list, np.dot(cov_matrix, weights_list)))
        sharpe = (port_return - self.risk_free_rate) / port_std
        
        # 추가 지표: 최대 가중치, 최소 가중치, 효율적 자산 수
        max_weight = max(weights.values())
        min_weight = min(weights.values())
        effective_assets = sum(1 for w in weights.values() if w > 0.01)
        
        return {
            'return': port_return,
            'risk': port_std,
            'sharpe': sharpe,
            'max_weight': max_weight,
            'min_weight': min_weight,
            'effective_assets': effective_assets
        }
