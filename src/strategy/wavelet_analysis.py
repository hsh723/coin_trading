from typing import Dict, List
from dataclasses import dataclass
import pywt
import numpy as np

@dataclass
class WaveletAnalysis:
    decomposition_levels: Dict[str, np.ndarray]
    trend_strength: float
    cycle_periods: List[float]
    noise_level: float

class WaveletAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'wavelet_type': 'db8',
            'max_level': 4,
            'min_period': 8
        }
        
    async def analyze_signal(self, price_data: np.ndarray) -> WaveletAnalysis:
        """웨이블릿 분석 실행"""
        # 웨이블릿 분해
        coeffs = pywt.wavedec(price_data, self.config['wavelet_type'], 
                             level=self.config['max_level'])
                             
        levels = {f"level_{i}": coef for i, coef in enumerate(coeffs)}
        
        return WaveletAnalysis(
            decomposition_levels=levels,
            trend_strength=self._calculate_trend_strength(coeffs),
            cycle_periods=self._identify_cycles(coeffs),
            noise_level=self._estimate_noise(coeffs[-1])
        )
