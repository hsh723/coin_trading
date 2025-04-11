import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TradeImpactMetrics:
    permanent_impact: float
    temporary_impact: float
    recovery_time: float
    impact_asymmetry: float

class TradeImpactAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'impact_window': 50,
            'recovery_threshold': 0.8
        }
        
    async def analyze_impact(self, trade_data: Dict, market_data: Dict) -> TradeImpactMetrics:
        """거래 영향도 분석"""
        perm_impact = self._calculate_permanent_impact(trade_data, market_data)
        temp_impact = self._calculate_temporary_impact(trade_data, market_data)
        
        return TradeImpactMetrics(
            permanent_impact=perm_impact,
            temporary_impact=temp_impact,
            recovery_time=self._calculate_recovery_time(market_data),
            impact_asymmetry=self._calculate_impact_asymmetry(perm_impact, temp_impact)
        )
