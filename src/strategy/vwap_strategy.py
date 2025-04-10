from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class VWAPSignal:
    signal_type: str
    vwap_price: float
    price_to_vwap: float
    volume_profile: Dict[str, float]
    deviation_percentage: float

class VWAPStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'lookback_window': '1D',
            'deviation_threshold': 0.02  # 2%
        }
        
    async def generate_signals(self, market_data: pd.DataFrame) -> VWAPSignal:
        """VWAP 기반 신호 생성"""
        vwap = self._calculate_vwap(market_data)
        current_price = market_data['close'].iloc[-1]
        deviation = (current_price - vwap) / vwap
        
        return VWAPSignal(
            signal_type=self._determine_signal(current_price, vwap),
            vwap_price=vwap,
            price_to_vwap=current_price/vwap,
            volume_profile=self._analyze_volume_profile(market_data),
            deviation_percentage=deviation * 100
        )
