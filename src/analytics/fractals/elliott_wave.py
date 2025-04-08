import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class WavePattern:
    wave_degree: str
    wave_points: List[float]
    confidence: float
    completion: float

class ElliottWaveAnalyzer:
    def __init__(self, min_wave_size: float = 0.02):
        self.min_wave_size = min_wave_size
        self.degrees = ['Grand Super Cycle', 'Super Cycle', 'Cycle', 'Primary']
        
    def analyze_waves(self, price_data: pd.Series) -> Dict[str, WavePattern]:
        """엘리어트 파동 분석"""
        waves = {}
        pivots = self._find_pivot_points(price_data)
        
        for degree in self.degrees:
            waves[degree] = self._identify_wave_pattern(
                price_data, 
                pivots, 
                degree
            )
            
        return waves
