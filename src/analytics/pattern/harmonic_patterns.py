import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class HarmonicPattern:
    pattern_type: str
    points: List[float]
    ratios: Dict[str, float]
    completion_price: float
    confidence: float

class HarmonicPatternDetector:
    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance
        self.patterns = {
            'GARTLEY': {'XA': 0.618, 'AB': 0.382, 'BC': 0.886, 'CD': 1.272},
            'BUTTERFLY': {'XA': 0.786, 'AB': 0.382, 'BC': 0.886, 'CD': 1.618},
            'BAT': {'XA': 0.886, 'AB': 0.382, 'BC': 0.886, 'CD': 2.0},
            'CRAB': {'XA': 0.886, 'AB': 0.382, 'BC': 0.886, 'CD': 3.618}
        }
        
    def find_patterns(self, price_data: pd.Series) -> List[HarmonicPattern]:
        """하모닉 패턴 탐지"""
        patterns = []
        swing_points = self._find_swing_points(price_data)
        
        for i in range(len(swing_points) - 4):
            pattern = self._identify_pattern(swing_points[i:i+5])
            if pattern:
                patterns.append(pattern)
                
        return patterns
