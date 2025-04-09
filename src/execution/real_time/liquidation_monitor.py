from typing import Dict, List
from dataclasses import dataclass

@dataclass
class LiquidationRisk:
    liquidation_price: float
    distance_to_liquidation: float
    margin_ratio: float
    warning_level: str

class LiquidationMonitor:
    def __init__(self, risk_thresholds: Dict[str, float]):
        self.risk_thresholds = risk_thresholds
        self.positions = {}
        
    async def monitor_liquidation_risks(self, 
                                      positions: Dict, 
                                      market_prices: Dict) -> Dict[str, LiquidationRisk]:
        """실시간 청산 위험 모니터링"""
        risk_metrics = {}
        for symbol, position in positions.items():
            if symbol in market_prices:
                risk = self._calculate_liquidation_risk(
                    position, market_prices[symbol]
                )
                risk_metrics[symbol] = risk
                await self._check_warning_levels(symbol, risk)
        return risk_metrics
