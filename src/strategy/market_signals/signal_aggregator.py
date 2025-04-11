from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class AggregatedSignal:
    composite_score: float
    signal_confidence: float
    signal_sources: Dict[str, float]
    trading_bias: str

class SignalAggregator:
    def __init__(self):
        self.signal_weights = {
            'technical': 0.4,
            'volume': 0.2,
            'sentiment': 0.2,
            'momentum': 0.2
        }
        
    async def aggregate_signals(self, signals: Dict[str, float]) -> AggregatedSignal:
        composite = sum(signals[k] * self.signal_weights[k] 
                       for k in self.signal_weights.keys())
        
        return AggregatedSignal(
            composite_score=composite,
            signal_confidence=self._calculate_confidence(signals),
            signal_sources=signals,
            trading_bias=self._determine_bias(composite)
        )
