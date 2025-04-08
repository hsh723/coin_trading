import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RiskDecomposition:
    systematic_risk: float
    specific_risk: float
    risk_contributions: Dict[str, float]
    factor_exposures: Dict[str, float]

class RiskDecompositionAnalyzer:
    def __init__(self, risk_factors: List[str]):
        self.risk_factors = risk_factors
        
    def decompose_risk(self, returns: pd.DataFrame, weights: np.ndarray) -> RiskDecomposition:
        """포트폴리오 리스크 분해"""
        factor_returns = self._calculate_factor_returns(returns)
        betas = self._calculate_factor_betas(returns, factor_returns)
        
        systematic_risk = self._calculate_systematic_risk(betas, factor_returns)
        specific_risk = self._calculate_specific_risk(returns, systematic_risk)
        
        return RiskDecomposition(
            systematic_risk=systematic_risk,
            specific_risk=specific_risk,
            risk_contributions=self._calculate_risk_contributions(returns, weights),
            factor_exposures=dict(zip(self.risk_factors, betas))
        )
