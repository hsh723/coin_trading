from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class ADXSignal:
    trend_strength: float
    di_positive: float
    di_negative: float
    adx_value: float
    signal_type: str

class ADXStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'adx_period': 14,
            'di_period': 14,
            'trend_threshold': 25
        }
        
    async def generate_signal(self, market_data: pd.DataFrame) -> ADXSignal:
        """ADX 신호 생성"""
        adx = self._calculate_adx(market_data)
        di_pos = self._calculate_di_positive(market_data)
        di_neg = self._calculate_di_negative(market_data)
        
        return ADXSignal(
            trend_strength=adx[-1],
            di_positive=di_pos[-1],
            di_negative=di_neg[-1],
            adx_value=adx[-1],
            signal_type=self._determine_signal(di_pos[-1], di_neg[-1], adx[-1])
        )
