import numpy as np
from typing import Dict

class VolumeHurstAnalyzer:
    def __init__(self, min_k: int = 2, max_k: int = 20):
        self.min_k = min_k
        self.max_k = max_k
        
    async def calculate_hurst(self, volume_data: np.ndarray) -> Dict:
        """거래량 허스트 지수 분석"""
        hurst_index = self._calculate_hurst_exponent(volume_data)
        persistence = self._analyze_persistence(hurst_index)
        
        return {
            'hurst_exponent': hurst_index,
            'persistence_type': persistence,
            'fractal_dimension': 2 - hurst_index,
            'market_efficiency': self._evaluate_efficiency(hurst_index)
        }
