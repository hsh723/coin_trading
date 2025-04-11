import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class QuoteImpactMetrics:
    price_impact: float
    quote_efficiency: float
    market_response: float
    impact_decay: List[float]

class QuoteImpactAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    async def analyze_quote_impact(self, market_data: Dict) -> QuoteImpactMetrics:
        """호가 영향도 분석"""
        price_changes = self._calculate_price_changes(market_data)
        quote_changes = self._analyze_quote_changes(market_data)
        
        return QuoteImpactMetrics(
            price_impact=self._calculate_price_impact(price_changes),
            quote_efficiency=self._calculate_quote_efficiency(quote_changes),
            market_response=self._calculate_market_response(price_changes, quote_changes),
            impact_decay=self._calculate_impact_decay(market_data)
        )
