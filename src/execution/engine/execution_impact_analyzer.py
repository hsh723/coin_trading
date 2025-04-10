from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ImpactAnalysis:
    price_impact: float
    volume_impact: float
    spread_impact: float
    recovery_time: float
    total_impact_score: float

class ExecutionImpactAnalyzer:
    def __init__(self, impact_config: Dict = None):
        self.config = impact_config or {
            'impact_window': 100,
            'recovery_threshold': 0.8
        }
        
    async def analyze_impact(self, execution_data: Dict, 
                           market_data: Dict) -> ImpactAnalysis:
        """실행 영향도 분석"""
        price_impact = self._calculate_price_impact(execution_data, market_data)
        volume_impact = self._calculate_volume_impact(execution_data, market_data)
        spread_impact = self._calculate_spread_impact(execution_data, market_data)
        recovery = self._estimate_recovery_time(price_impact, market_data)
        
        return ImpactAnalysis(
            price_impact=price_impact,
            volume_impact=volume_impact,
            spread_impact=spread_impact,
            recovery_time=recovery,
            total_impact_score=self._calculate_impact_score(
                price_impact, volume_impact, spread_impact
            )
        )
