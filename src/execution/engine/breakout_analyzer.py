from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class BreakoutAnalysis:
    breakout_level: float
    breakout_strength: float
    confirmation_signals: List[str]
    target_levels: List[float]
    stop_level: float

class BreakoutAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'confirmation_period': 3,
            'volume_threshold': 1.5,
            'strength_threshold': 0.02
        }
        
    async def analyze_breakout(self, price_data: np.ndarray, 
                             volume_data: np.ndarray) -> BreakoutAnalysis:
        """브레이크아웃 분석"""
        breakout_level = self._find_breakout_level(price_data)
        strength = self._calculate_breakout_strength(price_data, volume_data)
        
        return BreakoutAnalysis(
            breakout_level=breakout_level,
            breakout_strength=strength,
            confirmation_signals=self._get_confirmation_signals(price_data, volume_data),
            target_levels=self._calculate_target_levels(breakout_level, strength),
            stop_level=self._calculate_stop_level(breakout_level, strength)
        )
