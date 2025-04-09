from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class TickAnalysis:
    tick_direction: int
    price_momentum: float
    tick_volume: float
    micro_price: float
    spread: float

class TickAnalyzer:
    def __init__(self):
        self.last_tick = None
        
    async def analyze_tick(self, tick: Dict) -> TickAnalysis:
        """실시간 틱 분석"""
        direction = self._calculate_tick_direction(tick)
        momentum = self._calculate_momentum(tick)
        
        analysis = TickAnalysis(
            tick_direction=direction,
            price_momentum=momentum,
            tick_volume=tick['volume'],
            micro_price=self._calculate_micro_price(tick),
            spread=self._calculate_spread(tick)
        )
        
        self.last_tick = tick
        return analysis
