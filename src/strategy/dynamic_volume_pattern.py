from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class VolumePatternSignal:
    pattern_type: str
    pattern_strength: float
    volume_ratio: float
    price_correlation: float
    action_signal: str

class DynamicVolumePattern:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'pattern_window': 30,
            'volume_threshold': 1.5,
            'correlation_window': 20
        }
        
    async def analyze_volume_pattern(self, market_data: pd.DataFrame) -> VolumePatternSignal:
        """역동적 거래량 패턴 분석"""
        volume_series = market_data['volume']
        price_series = market_data['close']
        
        pattern = self._identify_volume_pattern(volume_series)
        strength = self._calculate_pattern_strength(volume_series)
        correlation = self._calculate_price_volume_correlation(price_series, volume_series)
        
        return VolumePatternSignal(
            pattern_type=pattern,
            pattern_strength=strength,
            volume_ratio=self._calculate_volume_ratio(volume_series),
            price_correlation=correlation,
            action_signal=self._generate_signal(pattern, strength, correlation)
        )
