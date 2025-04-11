import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeSpread:
    bid_volume: float
    ask_volume: float
    spread_ratio: float
    imbalance_score: float

class VolumeSpreadAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'imbalance_threshold': 0.2,
            'min_volume': 1.0
        }
        
    async def analyze_spread(self, order_book: Dict) -> VolumeSpread:
        """거래량 스프레드 분석"""
        bid_vol = self._calculate_bid_volume(order_book)
        ask_vol = self._calculate_ask_volume(order_book)
        
        return VolumeSpread(
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            spread_ratio=bid_vol / (bid_vol + ask_vol),
            imbalance_score=abs(bid_vol - ask_vol) / (bid_vol + ask_vol)
        )
