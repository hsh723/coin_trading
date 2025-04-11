from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class SpikeDetection:
    price_spike: bool
    volume_spike: bool
    spike_direction: str  # up, down
    impact_level: float
    confidence: float

class SpikeDetectionStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'price_threshold': 0.02,  # 2%
            'volume_threshold': 3.0,   # 3x normal volume
            'detection_window': 5      # 5 candles
        }
        
    async def detect_spikes(self, market_data: pd.DataFrame) -> SpikeDetection:
        """급격한 가격/거래량 변화 감지"""
        price_changes = market_data['close'].pct_change()
        volume_changes = market_data['volume'].pct_change()
        
        return SpikeDetection(
            price_spike=self._detect_price_spike(price_changes),
            volume_spike=self._detect_volume_spike(volume_changes),
            spike_direction=self._determine_spike_direction(price_changes),
            impact_level=self._calculate_impact(price_changes, volume_changes),
            confidence=self._calculate_spike_confidence(price_changes, volume_changes)
        )
