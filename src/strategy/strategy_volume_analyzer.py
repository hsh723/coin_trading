from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class VolumeAnalysisResult:
    volume_trend: str
    volume_momentum: float
    breakout_volume: bool
    volume_support_levels: List[float]

class StrategyVolumeAnalyzer:
    def __init__(self, analysis_config: Dict = None):
        self.config = analysis_config or {
            'volume_ma_periods': [20, 50],
            'breakout_threshold': 2.0,
            'support_lookback': 30
        }
        
    async def analyze_volume_patterns(self, 
                                    price_data: pd.DataFrame,
                                    volume_data: pd.Series) -> VolumeAnalysisResult:
        """거래량 패턴 분석"""
        trend = self._analyze_volume_trend(volume_data)
        momentum = self._calculate_volume_momentum(volume_data)
        breakout = self._detect_volume_breakout(volume_data)
        support = self._find_volume_support_levels(price_data, volume_data)
        
        return VolumeAnalysisResult(
            volume_trend=trend,
            volume_momentum=momentum,
            breakout_volume=breakout,
            volume_support_levels=support
        )
