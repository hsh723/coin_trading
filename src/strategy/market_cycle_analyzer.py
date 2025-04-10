from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketCycleSignal:
    cycle_phase: str  # accumulation, markup, distribution, markdown
    confidence: float
    support_levels: List[float]
    resistance_levels: List[float]
    cycle_duration: int

class MarketCycleAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_phase_length': 20,
            'cycle_lookback': 180,
            'volume_weight': 0.3
        }
        
    async def analyze_cycle(self, market_data: pd.DataFrame) -> MarketCycleSignal:
        """마켓 사이클 분석"""
        phase = self._identify_cycle_phase(market_data)
        levels = self._identify_key_levels(market_data)
        
        return MarketCycleSignal(
            cycle_phase=phase,
            confidence=self._calculate_phase_confidence(market_data, phase),
            support_levels=levels['support'],
            resistance_levels=levels['resistance'],
            cycle_duration=self._estimate_cycle_duration(market_data)
        )
