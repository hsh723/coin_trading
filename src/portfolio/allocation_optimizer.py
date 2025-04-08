import numpy as np
from scipy.optimize import minimize
from typing import Dict, List

class AllocationOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def optimize_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """최적 자산 배분 비율 계산"""
        n_assets = len(returns.columns)
        
        def objective(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe_ratio
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        result = minimize(
            objective,
            x0=np.array([1/n_assets] * n_assets),
            method='SLSQP',
            constraints=constraints
        )
        
        return dict(zip(returns.columns, result.x))
