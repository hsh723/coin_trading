from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class EMACrossSignal:
    signal_type: str
    fast_ema: float
    slow_ema: float
    cross_value: float
    trend_strength: float

class EMACrossStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'fast_period': 12,
            'slow_period': 26,
            'signal_threshold': 0.0002
        }
        
    async def generate_signal(self, market_data: pd.DataFrame) -> EMACrossSignal:
        """EMA 크로스오버 신호 생성"""
        fast_ema = self._calculate_ema(market_data['close'], self.config['fast_period'])
        slow_ema = self._calculate_ema(market_data['close'], self.config['slow_period'])
        
        return EMACrossSignal(
            signal_type=self._determine_signal(fast_ema[-1], slow_ema[-1]),
            fast_ema=fast_ema[-1],
            slow_ema=slow_ema[-1],
            cross_value=fast_ema[-1] - slow_ema[-1],
            trend_strength=self._calculate_trend_strength(fast_ema, slow_ema)
        )
