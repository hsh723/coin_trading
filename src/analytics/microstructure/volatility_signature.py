import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolatilitySignature:
    realized_volatility: float
    signature_plot: List[float]
    noise_ratio: float
    optimal_sampling: int

class VolatilitySignatureAnalyzer:
    def __init__(self, max_scale: int = 20):
        self.max_scale = max_scale
        
    async def analyze_signature(self, returns: np.ndarray) -> VolatilitySignature:
        """변동성 시그니처 분석"""
        scales = range(1, self.max_scale + 1)
        signature = [self._calculate_volatility(returns, scale) for scale in scales]
        
        return VolatilitySignature(
            realized_volatility=signature[0],
            signature_plot=signature,
            noise_ratio=self._calculate_noise_ratio(signature),
            optimal_sampling=self._find_optimal_sampling(signature)
        )
