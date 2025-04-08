import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SpreadMetrics:
    avg_spread: float
    spread_volatility: float
    spread_percentiles: Dict[str, float]
    cost_impact: float

class SpreadAnalyzer:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        
    def analyze_spreads(self, orderbook_data: pd.DataFrame) -> SpreadMetrics:
        """스프레드 분석"""
        spreads = orderbook_data['ask'] - orderbook_data['bid']
        
        return SpreadMetrics(
            avg_spread=spreads.mean(),
            spread_volatility=spreads.std(),
            spread_percentiles={
                '25': spreads.quantile(0.25),
                '50': spreads.quantile(0.50),
                '75': spreads.quantile(0.75)
            },
            cost_impact=self._calculate_cost_impact(spreads)
        )
