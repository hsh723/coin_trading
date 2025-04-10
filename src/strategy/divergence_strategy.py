from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class DivergenceSignal:
    divergence_type: str  # regular, hidden, positive, negative
    indicator_name: str   # RSI, MACD, etc
    strength: float
    price_points: List[float]
    signal_type: str      # buy, sell, hold

class DivergenceStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'indicators': ['rsi', 'macd'],
            'min_divergence_strength': 0.7,
            'confirmation_period': 3
        }
        
    async def find_divergences(self, market_data: pd.DataFrame) -> List[DivergenceSignal]:
        """다이버전스 패턴 탐색"""
        signals = []
        
        for indicator in self.config['indicators']:
            if divergence := self._check_divergence(market_data, indicator):
                signals.append(divergence)
                
        return sorted(signals, key=lambda x: x.strength, reverse=True)
