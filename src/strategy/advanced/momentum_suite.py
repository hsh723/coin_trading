from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MomentumSignals:
    rsi_signal: float
    macd_signal: Dict[str, float]
    adx_signal: float
    combined_score: float

class MomentumSuite:
    def __init__(self):
        self.indicators = {
            'rsi': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'adx': {'period': 14}
        }
        
    async def generate_signals(self, data: pd.DataFrame) -> MomentumSignals:
        rsi = self._calculate_rsi(data)
        macd = self._calculate_macd(data)
        adx = self._calculate_adx(data)
        
        return MomentumSignals(
            rsi_signal=rsi,
            macd_signal=macd,
            adx_signal=adx,
            combined_score=self._combine_signals(rsi, macd, adx)
        )
