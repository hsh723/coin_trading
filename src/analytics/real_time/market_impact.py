from typing import Dict, List
import numpy as np

class MarketImpactAnalyzer:
    def __init__(self, impact_window: int = 100):
        self.impact_window = impact_window
        self.impact_history = []
        
    async def analyze_impact(self, trade_data: Dict, market_data: Dict) -> Dict:
        """실시간 시장 영향도 분석"""
        price_impact = self._calculate_price_impact(trade_data, market_data)
        volume_impact = self._calculate_volume_impact(trade_data)
        liquidity_impact = self._analyze_liquidity_impact(market_data)
        
        return {
            'price_impact': price_impact,
            'volume_impact': volume_impact,
            'liquidity_impact': liquidity_impact,
            'total_impact': self._combine_impacts(price_impact, volume_impact, liquidity_impact)
        }
