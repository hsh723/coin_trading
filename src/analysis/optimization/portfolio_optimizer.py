import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple

class PortfolioOptimizer:
    def __init__(self, constraints: Dict = None):
        self.constraints = constraints or {
            'min_weight': 0.0,
            'max_weight': 0.3
        }
        
    def optimize_maximum_sharpe(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
        """샤프 비율 최대화 포트폴리오"""
        n_assets = len(returns.columns)
        bounds = tuple((self.constraints['min_weight'], self.constraints['max_weight']) 
                      for _ in range(n_assets))
                      
        result = minimize(
            lambda w: -self._calculate_sharpe_ratio(w, returns, risk_free_rate),
            x0=np.array([1/n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        return dict(zip(returns.columns, result.x))
