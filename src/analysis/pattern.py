import numpy as np
import talib
from typing import Dict, List

class PatternRecognizer:
    def __init__(self):
        self.patterns = {
            'DOJI': talib.CDLDOJI,
            'HAMMER': talib.CDLHAMMER,
            'ENGULFING': talib.CDLENGULFING,
            'MORNING_STAR': talib.CDLMORNINGSTAR
        }
    
    def identify_patterns(self, ohlc_data: Dict) -> List[str]:
        """주요 캔들스틱 패턴 식별"""
        patterns_found = []
        for pattern_name, pattern_func in self.patterns.items():
            result = pattern_func(
                ohlc_data['open'],
                ohlc_data['high'],
                ohlc_data['low'],
                ohlc_data['close']
            )
            if np.any(result != 0):
                patterns_found.append(pattern_name)
        return patterns_found
