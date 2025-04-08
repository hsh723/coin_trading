from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class BarPattern:
    pattern_type: str
    strength: float
    bars_involved: int
    significance: float

class BarPatternAnalyzer:
    def __init__(self, lookback_period: int = 5):
        self.lookback_period = lookback_period
        self.patterns = {
            'engulfing': self._check_engulfing,
            'hammer': self._check_hammer,
            'shooting_star': self._check_shooting_star,
            'doji': self._check_doji
        }
        
    def analyze_patterns(self, ohlc_data: pd.DataFrame) -> List[BarPattern]:
        """캔들스틱 패턴 분석"""
        patterns = []
        for i in range(len(ohlc_data) - self.lookback_period, len(ohlc_data)):
            window = ohlc_data.iloc[i-2:i+1]
            for pattern_name, check_func in self.patterns.items():
                if pattern := check_func(window):
                    patterns.append(pattern)
        return patterns
