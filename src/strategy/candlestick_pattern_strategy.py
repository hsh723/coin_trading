from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class CandlePattern:
    pattern_name: str
    strength: float
    confirmation: bool
    support_resistance: Dict[str, float]

class CandlestickPatternStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_pattern_size': 3,
            'confirmation_period': 2,
            'strength_threshold': 0.7
        }
        
    async def identify_patterns(self, market_data: pd.DataFrame) -> List[CandlePattern]:
        """캔들스틱 패턴 식별"""
        patterns = []
        
        if self._is_doji(market_data):
            patterns.append(self._create_doji_pattern(market_data))
            
        if self._is_hammer(market_data):
            patterns.append(self._create_hammer_pattern(market_data))
            
        if self._is_engulfing(market_data):
            patterns.append(self._create_engulfing_pattern(market_data))
            
        return sorted(patterns, key=lambda x: x.strength, reverse=True)
