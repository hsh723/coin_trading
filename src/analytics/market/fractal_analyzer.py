import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class FractalAnalysis:
    fractal_dimension: float
    pattern_similarity: float
    scale_invariance: Dict[str, float]
    fractal_levels: List[float]

class FractalAnalyzer:
    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [10, 20, 50, 100]
        
    async def analyze_fractals(self, price_data: np.ndarray) -> FractalAnalysis:
        """프랙탈 분석"""
        dimension = self._calculate_fractal_dimension(price_data)
        similarity = self._calculate_pattern_similarity(price_data)
        
        return FractalAnalysis(
            fractal_dimension=dimension,
            pattern_similarity=similarity,
            scale_invariance=self._analyze_scale_invariance(price_data),
            fractal_levels=self._identify_fractal_levels(price_data)
        )
