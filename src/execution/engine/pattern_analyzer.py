from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class PatternAnalysis:
    pattern_type: str
    confidence: float
    price_target: float
    support_resistance: List[float]
    completion_percent: float

class PatternAnalyzer:
    def __init__(self, min_pattern_size: int = 5):
        self.min_pattern_size = min_pattern_size
        self.patterns = {
            'head_and_shoulders': self._check_head_shoulders,
            'double_top': self._check_double_top,
            'double_bottom': self._check_double_bottom,
            'triangle': self._check_triangle
        }
        
    async def analyze_patterns(self, price_data: np.ndarray) -> PatternAnalysis:
        """차트 패턴 분석"""
        results = []
        for pattern_name, pattern_func in self.patterns.items():
            if pattern_func(price_data):
                results.append((pattern_name, self._calculate_confidence(price_data)))
                
        if not results:
            return None
            
        best_pattern = max(results, key=lambda x: x[1])
        return PatternAnalysis(
            pattern_type=best_pattern[0],
            confidence=best_pattern[1],
            price_target=self._calculate_target(price_data, best_pattern[0]),
            support_resistance=self._find_support_resistance(price_data),
            completion_percent=self._calculate_completion(price_data, best_pattern[0])
        )
