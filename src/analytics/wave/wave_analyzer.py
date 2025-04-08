import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class WavePattern:
    wave_count: int
    current_wave: int
    wave_points: List[Dict]
    pattern_strength: float

class WaveAnalyzer:
    def __init__(self, min_wave_size: float = 0.01):
        self.min_wave_size = min_wave_size
        
    def analyze_waves(self, price_data: pd.Series) -> WavePattern:
        """가격 파동 분석"""
        waves = self._identify_waves(price_data)
        wave_count = len(waves)
        
        return WavePattern(
            wave_count=wave_count,
            current_wave=self._determine_current_wave(waves),
            wave_points=waves,
            pattern_strength=self._calculate_pattern_strength(waves)
        )
