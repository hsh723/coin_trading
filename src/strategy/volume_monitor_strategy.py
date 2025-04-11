from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class VolumeMonitorSignal:
    volume_surge: bool
    distribution_pattern: str
    accumulation_zones: List[float]
    volume_profile_score: float

class VolumeMonitorStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'surge_threshold': 2.0,
            'analysis_window': 20,
            'min_zone_strength': 0.7
        }
        
    async def analyze_volume_patterns(self, market_data: pd.DataFrame) -> VolumeMonitorSignal:
        """거래량 패턴 분석"""
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(
            window=self.config['analysis_window']
        ).mean().iloc[-1]
        
        return VolumeMonitorSignal(
            volume_surge=current_volume > avg_volume * self.config['surge_threshold'],
            distribution_pattern=self._identify_distribution_pattern(market_data),
            accumulation_zones=self._find_accumulation_zones(market_data),
            volume_profile_score=self._calculate_volume_score(market_data)
        )
