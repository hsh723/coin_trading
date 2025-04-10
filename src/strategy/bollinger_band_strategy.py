from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class BollingerSignal:
    signal_type: str
    band_position: float
    band_width: float
    price_trend: str
    volatility: float

class BollingerBandStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'period': 20,
            'std_dev': 2,
            'squeeze_threshold': 0.1
        }
        
    async def generate_signals(self, market_data: pd.DataFrame) -> BollingerSignal:
        """볼린저 밴드 신호 생성"""
        bb_data = self._calculate_bollinger_bands(market_data['close'])
        current_price = market_data['close'].iloc[-1]
        
        return BollingerSignal(
            signal_type=self._determine_signal(current_price, bb_data),
            band_position=self._calculate_band_position(current_price, bb_data),
            band_width=self._calculate_band_width(bb_data),
            price_trend=self._determine_trend(market_data),
            volatility=self._calculate_volatility(bb_data)
        )
