import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PivotLevels:
    pivot: float
    support_levels: List[float]
    resistance_levels: List[float]
    projected_range: Dict[str, float]

class PivotPointAnalyzer:
    def __init__(self, method: str = 'standard'):
        self.method = method
        
    def calculate_pivot_points(self, high: float, low: float, close: float) -> PivotLevels:
        """피봇 포인트 및 지지/저항 레벨 계산"""
        pivot = (high + low + close) / 3
        
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        return PivotLevels(
            pivot=pivot,
            support_levels=[s1, s2, s3],
            resistance_levels=[r1, r2, r3],
            projected_range={'low': s1, 'high': r1}
        )
