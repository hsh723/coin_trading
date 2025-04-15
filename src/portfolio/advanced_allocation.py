import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.optimize import minimize

class RiskParityAllocator:
    """리스크 패리티 자산 배분
    
    포트폴리오의 리스크 기여도를 모든 자산에 대해 동일하게 분배하는 전략
    - 변동성 큰 자산에 적은 가중치 배분
    - 자산 간 상관관계 고려
    - 분산 효과 극대화
    """
    
    def __init__(self, risk_target: Optional[float] = None):
        """
        Args:
            risk_target: 포트폴리오 목표 리스크 (None이면 최적화로 결정)
        """
        self.risk_target = risk_target
        self.weights = None
        self.risk_contributions = None
    
    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        """리스크 패리티 가중치 계산
        
        Args:
            returns: 자산 수익률 데이터프레임
            
        Returns:
            자산 가중치 딕셔너리
        """
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values * 252  # 연간화된 공분산 행렬
        
        # 목적 함수: 리스크 기여도 편차 최소화
        def objective(weights):
            weights = np.array(weights)
            port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            port_volatility = np.sqrt(port_variance)
            
            # 각 자산 리스크 기여도
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_volatility
            
            # 목표 기여도: 각 자산이 동일한 리스크 기여
            target_risk_contrib = port_volatility / n_assets
            
            # 편차 제곱합 (MSE)
            risk_diff = np.sum((risk_contrib - target_risk_contrib)**2)
            return risk_diff
        
        # 제약 조건: 가중치 합 = 1, 가중치 > 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # 변수 범위: 0 <= 가중치 <= 1
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        
        # 초기 가중치: 균등 배분
        initial_weights = np.ones(n_assets) / n_assets
        
        # 최적화 실행
        result = minimize(
            objective, initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result['success']:
            raise ValueError(f"최적화 실패: {result['message']}")
        
        # 결과 저장
        weights = result['x']
        
        # 계산된 포트폴리오 리스크
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_volatility = np.sqrt(port_variance)
        
        # 각 자산 리스크 기여도
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / port_volatility
        
        # 리스크 기여 비율
        risk_contrib_pct = risk_contrib / port_volatility
        
        # 리스크 비율과 가중치 비율의 비
        risk_to_weight_ratio = risk_contrib_pct / weights
        
        # 결과를 딕셔너리로 변환
        self.weights = dict(zip(returns.columns, weights))
        self.risk_contributions = dict(zip(returns.columns, risk_contrib))
        
        return self.weights
    
    def get_risk_contributions(self) -> Dict[str, float]:
        """각 자산의 리스크 기여도 반환
        
        Returns:
            자산별 리스크 기여도 딕셔너리
        """
        if self.risk_contributions is None:
            raise ValueError("먼저 allocate 메서드를 호출하세요")
        return self.risk_contributions
    
    def get_equal_risk_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """단순화된 방법으로 리스크 패리티 가중치 계산
        
        상관관계를 무시하고 변동성의 역수에 비례하여 가중치 계산
        
        Args:
            returns: 자산 수익률 데이터프레임
            
        Returns:
            간소화된 리스크 패리티 가중치
        """
        # 각 자산 변동성 계산
        vols = returns.std() * np.sqrt(252)
        
        # 변동성 역수에 비례하는 가중치 계산
        inverse_vols = 1 / vols
        weights = inverse_vols / inverse_vols.sum()
        
        return dict(zip(returns.columns, weights))


class KellyCriterionAllocator:
    """켈리 크라이테리온 기반 자산 배분
    
    장기적인 자본 성장을 극대화하는 최적 배분 비율 계산
    - 수학적으로 장기 자본 성장률 극대화
    - 수익률과 리스크의 균형 추구
    - 분산 투자 효과를 고려한 다차원 켈리 공식 구현
    """
    
    def __init__(self, fraction: float = 1.0):
        """
        Args:
            fraction: 켈리 비율 조정 (1.0: 전체 켈리, 0.5: 하프 켈리)
        """
        self.fraction = fraction
        self.weights = None
    
    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        """켈리 크라이테리온 기반 가중치 계산
        
        Args:
            returns: 자산 수익률 데이터프레임
            
        Returns:
            자산 가중치 딕셔너리
        """
        # 평균 수익률 및 공분산 행렬 계산
        mean_returns = returns.mean() * 252  # 연간화된 평균 수익률
        cov_matrix = returns.cov() * 252     # 연간화된 공분산 행렬
        
        # 켈리 공식: weights = inv(Σ) * μ
        # 여기서 Σ는 공분산 행렬, μ는 평균 초과 수익률
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            kelly_weights = np.dot(inv_cov, mean_returns)
            
            # 가중치 합이 1이 되도록 정규화
            kelly_weights = kelly_weights / np.sum(kelly_weights)
            
            # 켈리 비율 조정
            if self.fraction < 1.0:
                # 하프 켈리 등 비율 조정
                kelly_weights = kelly_weights * self.fraction
                
                # 남은 비율은 무위험 자산(현금)에 배분하지만,
                # 이 구현에서는 단순히 정규화만 수행
                kelly_weights = kelly_weights / np.sum(kelly_weights)
            
            # 음수 가중치 처리 (공매도 허용하지 않는 경우)
            if np.any(kelly_weights < 0):
                # 음수 가중치를 0으로 설정하고 재정규화하는 방식
                kelly_weights = np.maximum(kelly_weights, 0)
                kelly_weights = kelly_weights / np.sum(kelly_weights)
            
            self.weights = dict(zip(returns.columns, kelly_weights))
            return self.weights
            
        except np.linalg.LinAlgError:
            # 공분산 행렬이 특이행렬인 경우
            raise ValueError("공분산 행렬이 특이행렬입니다. 데이터를 확인하세요.")
    
    def get_fractional_kelly(self, returns: pd.DataFrame, 
                           fractions: List[float]) -> Dict[str, Dict[str, float]]:
        """다양한 켈리 비율에 대한 가중치 계산
        
        Args:
            returns: 자산 수익률 데이터프레임
            fractions: 켈리 비율 목록 (예: [0.25, 0.5, 0.75, 1.0])
            
        Returns:
            비율별 가중치 딕셔너리
        """
        result = {}
        original_fraction = self.fraction
        
        for f in fractions:
            self.fraction = f
            weights = self.allocate(returns)
            result[f] = weights.copy()
        
        # 원래 비율 복원
        self.fraction = original_fraction
        return result


class ConstantProportionPortfolio:
    """일정 비율 포트폴리오 관리 (CPPI)
    
    하방 리스크 보호와 상승 참여를 동시에 추구하는 동적 자산 배분 전략
    - 원금 보존을 위한 최소 쿠션 유지
    - 시장 상황에 따른 동적 레버리지 조정
    - 옵션과 유사한 비대칭적 수익 프로파일 구축
    """
    
    def __init__(self, floor_percentage: float = 0.8, 
                multiplier: float = 3.0,
                max_leverage: float = 2.0):
        """
        Args:
            floor_percentage: 보호 레벨 (원금의 %)
            multiplier: 쿠션 승수
            max_leverage: 최대 레버리지 비율
        """
        self.floor_percentage = floor_percentage
        self.multiplier = multiplier
        self.max_leverage = max_leverage
    
    def allocate(self, portfolio_value: float, 
               risky_asset_allocation: Dict[str, float]) -> Dict[str, float]:
        """CPPI 전략으로 위험 자산 배분 계산
        
        Args:
            portfolio_value: 포트폴리오 총 가치
            risky_asset_allocation: 위험 자산 내부 배분 비율
            
        Returns:
            전체 자산 배분 가중치 (안전 자산 포함)
        """
        # 최소 보존 금액 (floor)
        floor_value = portfolio_value * self.floor_percentage
        
        # 쿠션 (투자 가능 금액)
        cushion = portfolio_value - floor_value
        
        # 위험 자산 배분 금액
        risky_allocation = cushion * self.multiplier
        
        # 최대 레버리지 제한
        risky_allocation = min(risky_allocation, portfolio_value * self.max_leverage)
        
        # 안전 자산 배분 금액
        safe_allocation = portfolio_value - risky_allocation
        
        # 가중치 계산
        risky_weight = risky_allocation / portfolio_value
        safe_weight = safe_allocation / portfolio_value
        
        # 전체 가중치 계산
        weights = {'Cash': safe_weight}  # 현금 또는 안전 자산
        
        # 위험 자산 내부 비율 적용
        for asset, alloc in risky_asset_allocation.items():
            weights[asset] = risky_weight * alloc
        
        return weights
    
    def simulate(self, prices: pd.DataFrame, 
                initial_value: float = 10000,
                risky_asset_weights: Dict[str, float] = None) -> pd.DataFrame:
        """CPPI 전략 시뮬레이션
        
        Args:
            prices: 자산 가격 데이터프레임
            initial_value: 초기 포트폴리오 가치
            risky_asset_weights: 위험 자산 내부 배분 비율
            
        Returns:
            시뮬레이션 결과 데이터프레임
        """
        # 위험 자산 내부 배분 설정
        if risky_asset_weights is None:
            # 기본값: 모든 자산 균등 배분
            risky_asset_weights = {col: 1.0/len(prices.columns) for col in prices.columns}
        
        # 결과 저장 데이터프레임
        results = pd.DataFrame(index=prices.index)
        results['portfolio_value'] = np.nan
        results['risky_allocation'] = np.nan
        results['safe_allocation'] = np.nan
        results['risky_weight'] = np.nan
        results['floor_value'] = np.nan
        
        # 초기 자산 배분
        portfolio_value = initial_value
        
        # 각 시점에 대해 시뮬레이션
        for i, date in enumerate(prices.index):
            # 첫 날은 초기 배분
            if i == 0:
                # 최소 보존 금액
                floor_value = portfolio_value * self.floor_percentage
                
                # 쿠션
                cushion = portfolio_value - floor_value
                
                # 위험 자산 배분
                risky_allocation = min(cushion * self.multiplier, 
                                     portfolio_value * self.max_leverage)
                
                # 안전 자산 배분
                safe_allocation = portfolio_value - risky_allocation
                
                # 위험 자산 비중
                risky_weight = risky_allocation / portfolio_value
                
                # 결과 저장
                results.loc[date, 'portfolio_value'] = portfolio_value
                results.loc[date, 'risky_allocation'] = risky_allocation
                results.loc[date, 'safe_allocation'] = safe_allocation
                results.loc[date, 'risky_weight'] = risky_weight
                results.loc[date, 'floor_value'] = floor_value
                
                # 자산별 투자 금액 저장
                for asset in prices.columns:
                    results.loc[date, f'amount_{asset}'] = (
                        risky_allocation * risky_asset_weights[asset]
                    )
                    results.loc[date, f'weight_{asset}'] = (
                        risky_weight * risky_asset_weights[asset]
                    )
                
                # 각 자산별 보유 수량 계산
                asset_quantities = {}
                for asset in prices.columns:
                    asset_allocation = risky_allocation * risky_asset_weights[asset]
                    asset_quantities[asset] = asset_allocation / prices.loc[date, asset]
                
                # 안전 자산 금액
                cash_amount = safe_allocation
                
            else:
                # 위험 자산 가치 업데이트
                risky_value = 0
                for asset, quantity in asset_quantities.items():
                    risky_value += quantity * prices.loc[date, asset]
                
                # 총 포트폴리오 가치
                portfolio_value = risky_value + cash_amount
                
                # 최소 보존 금액 업데이트
                floor_value = portfolio_value * self.floor_percentage
                
                # 쿠션 업데이트
                cushion = portfolio_value - floor_value
                
                # 위험 자산 배분 업데이트
                target_risky_allocation = min(cushion * self.multiplier, 
                                            portfolio_value * self.max_leverage)
                
                # 재조정
                if i % 5 == 0:  # 5일마다 재조정 (예시)
                    # 현재 위험 자산 가치와 목표 가치의 차이
                    allocation_diff = target_risky_allocation - risky_value
                    
                    if abs(allocation_diff) > portfolio_value * 0.05:  # 5% 이상 차이 시 재조정
                        # 위험 자산 재조정
                        for asset in asset_quantities.keys():
                            # 새로운 자산 배분 금액
                            new_asset_allocation = (
                                target_risky_allocation * risky_asset_weights[asset]
                            )
                            
                            # 새로운 수량
                            asset_quantities[asset] = (
                                new_asset_allocation / prices.loc[date, asset]
                            )
                        
                        # 안전 자산 금액 업데이트
                        cash_amount = portfolio_value - target_risky_allocation
                
                # 위험 자산 비중
                risky_weight = risky_value / portfolio_value
                
                # 결과 저장
                results.loc[date, 'portfolio_value'] = portfolio_value
                results.loc[date, 'risky_allocation'] = risky_value
                results.loc[date, 'safe_allocation'] = cash_amount
                results.loc[date, 'risky_weight'] = risky_weight
                results.loc[date, 'floor_value'] = floor_value
                
                # 자산별 투자 금액 및 비중 저장
                for asset in prices.columns:
                    asset_value = asset_quantities[asset] * prices.loc[date, asset]
                    results.loc[date, f'amount_{asset}'] = asset_value
                    results.loc[date, f'weight_{asset}'] = asset_value / portfolio_value
        
        # 수익률 계산
        results['return'] = results['portfolio_value'].pct_change()
        results['cumulative_return'] = (1 + results['return']).cumprod() - 1
        
        return results


class MinimumCorrelationAllocator:
    """최소 상관계수 기반 자산 배분
    
    자산 간 상관관계를 최소화하여 분산 효과를 극대화하는 전략
    - 자산 간 상관관계 최소화
    - 변동성 고려 없이 분산 효과만 추구
    - 비가중 최소 상관관계 포트폴리오
    """
    
    def __init__(self):
        self.weights = None
    
    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        """최소 상관계수 가중치 계산
        
        Args:
            returns: 자산 수익률 데이터프레임
            
        Returns:
            자산 가중치 딕셔너리
        """
        # 상관계수 행렬
        corr_matrix = returns.corr().values
        
        # 상관계수 행렬의 역수 계산 (비상관 측정)
        np.fill_diagonal(corr_matrix, 1)  # 대각선 요소 1로 설정
        inv_corr = np.linalg.inv(corr_matrix)
        
        # 각 행의 합계 계산
        row_sums = np.sum(inv_corr, axis=1)
        
        # 가중치 계산
        weights = row_sums / np.sum(row_sums)
        
        # 결과를 딕셔너리로 변환
        self.weights = dict(zip(returns.columns, weights))
        return self.weights 