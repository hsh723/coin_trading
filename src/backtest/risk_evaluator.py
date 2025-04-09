import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    var: float
    expected_shortfall: float
    max_drawdown: float
    beta: float
    correlation_matrix: np.ndarray

class RiskEvaluator:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def evaluate_risks(self, 
                      returns: np.ndarray, 
                      benchmark_returns: np.ndarray = None) -> RiskMetrics:
        """백테스트 결과의 리스크 평가"""
        sorted_returns = np.sort(returns)
        var_idx = int((1 - self.confidence_level) * len(returns))
        
        return RiskMetrics(
            var=sorted_returns[var_idx],
            expected_shortfall=self._calculate_expected_shortfall(sorted_returns, var_idx),
            max_drawdown=self._calculate_max_drawdown(returns),
            beta=self._calculate_beta(returns, benchmark_returns),
            correlation_matrix=np.corrcoef(returns, benchmark_returns)
        )
