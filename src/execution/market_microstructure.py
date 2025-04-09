import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class MicrostructureMetrics:
    effective_spread: float
    realized_spread: float
    price_impact: float
    trade_size_dist: Dict[str, float]

class MarketMicrostructureAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    def analyze_microstructure(self, trades: pd.DataFrame,
                             orderbook: pd.DataFrame) -> MicrostructureMetrics:
        """시장 미시구조 분석"""
        effective_spread = self._calculate_effective_spread(orderbook)
        realized_spread = self._calculate_realized_spread(trades)
        
        return MicrostructureMetrics(
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            price_impact=self._calculate_price_impact(trades),
            trade_size_dist=self._analyze_trade_size_distribution(trades)
        )
