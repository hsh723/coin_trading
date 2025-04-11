from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class TimeframeAnalysis:
    primary_trend: str
    secondary_trend: str
    alignment_score: float
    confirmation_signals: Dict[str, bool]

class MultiTimeframeAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'timeframes': ['1h', '4h', '1d'],
            'alignment_threshold': 0.7
        }
        
    async def analyze_timeframes(self, 
                               market_data: Dict[str, pd.DataFrame]) -> TimeframeAnalysis:
        """멀티타임프레임 분석"""
        trends = {
            tf: self._analyze_single_timeframe(data)
            for tf, data in market_data.items()
        }
        
        return TimeframeAnalysis(
            primary_trend=self._get_primary_trend(trends),
            secondary_trend=self._get_secondary_trend(trends),
            alignment_score=self._calculate_alignment(trends),
            confirmation_signals=self._get_confirmations(trends)
        )
