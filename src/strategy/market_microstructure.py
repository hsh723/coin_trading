from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class MicrostructureSignal:
    effective_spread: float
    market_impact: float
    order_imbalance: float
    tick_direction: str
    execution_quality: float

class MarketMicrostructure:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'spread_window': 100,
            'impact_threshold': 0.001,
            'tick_window': 50
        }
        
    async def analyze_microstructure(self, 
                                   trade_data: pd.DataFrame,
                                   order_book: pd.DataFrame) -> MicrostructureSignal:
        """시장 미시구조 분석"""
        spread = self._calculate_effective_spread(order_book)
        impact = self._estimate_market_impact(trade_data)
        
        return MicrostructureSignal(
            effective_spread=spread,
            market_impact=impact,
            order_imbalance=self._calculate_order_imbalance(order_book),
            tick_direction=self._analyze_tick_direction(trade_data),
            execution_quality=self._evaluate_execution_quality(spread, impact)
        )
