from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class MarketAnalysis:
    market_regime: str
    trend_strength: float
    support_resistance: Dict[str, float]
    key_levels: List[float]
    volatility_state: Dict[str, float]

class StrategyMarketAnalyzer:
    def __init__(self, analysis_config: Dict = None):
        self.config = analysis_config or {
            'ma_periods': [20, 50, 200],
            'volatility_window': 20,
            'trend_threshold': 0.02
        }
        
    async def analyze_market(self, market_data: pd.DataFrame) -> MarketAnalysis:
        """시장 상태 분석"""
        trend = self._analyze_trend(market_data)
        volatility = self._analyze_volatility(market_data)
        levels = self._find_key_levels(market_data)
        
        return MarketAnalysis(
            market_regime=self._determine_regime(trend, volatility),
            trend_strength=trend['strength'],
            support_resistance=levels['support_resistance'],
            key_levels=levels['key_levels'],
            volatility_state=volatility
        )
