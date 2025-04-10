from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class IntermarketSignal:
    correlation_matrix: Dict[str, Dict[str, float]]
    leading_markets: List[str]
    signal_type: str
    confidence: float

class IntermarketAnalysis:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'correlation_threshold': 0.7,
            'lookback_period': 30,
            'markets': ['BTC', 'ETH', 'DXY', 'SPX']
        }
        
    async def analyze_markets(self, market_data: Dict[str, pd.DataFrame]) -> IntermarketSignal:
        """시장간 관계 분석"""
        correlations = self._calculate_correlations(market_data)
        leaders = self._identify_leading_markets(market_data)
        
        return IntermarketSignal(
            correlation_matrix=correlations,
            leading_markets=leaders,
            signal_type=self._determine_signal(correlations, leaders),
            confidence=self._calculate_confidence(correlations)
        )
