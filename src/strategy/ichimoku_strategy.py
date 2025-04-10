from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class IchimokuSignal:
    signal_type: str
    cloud_status: str
    trend_strength: float
    support_resistance: Dict[str, float]

class IchimokuStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_period': 52
        }
        
    async def generate_signals(self, market_data: pd.DataFrame) -> IchimokuSignal:
        """일목균형표 신호 생성"""
        ichimoku = self._calculate_ichimoku(market_data)
        current_price = market_data['close'].iloc[-1]
        
        return IchimokuSignal(
            signal_type=self._determine_signal(ichimoku, current_price),
            cloud_status=self._get_cloud_status(ichimoku, current_price),
            trend_strength=self._calculate_trend_strength(ichimoku),
            support_resistance=self._find_key_levels(ichimoku)
        )
