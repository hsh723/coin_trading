import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TimeframeAnalysis:
    dominant_trend: str
    trend_alignment: float
    timeframe_signals: Dict[str, Dict]
    conflict_levels: List[float]

class MultiTimeframeAnalyzer:
    def __init__(self, timeframes: List[str] = None):
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h', '1d']
        
    async def analyze_timeframes(self, market_data: Dict[str, pd.DataFrame]) -> TimeframeAnalysis:
        """다중 타임프레임 분석"""
        signals = {tf: self._analyze_single_timeframe(data) 
                  for tf, data in market_data.items()}
                  
        return TimeframeAnalysis(
            dominant_trend=self._find_dominant_trend(signals),
            trend_alignment=self._calculate_alignment(signals),
            timeframe_signals=signals,
            conflict_levels=self._identify_conflict_levels(signals)
        )
