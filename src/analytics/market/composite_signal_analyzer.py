import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CompositeSignal:
    signal_strength: float
    signal_direction: str
    confidence_score: float
    component_signals: Dict[str, float]

class CompositeSignalAnalyzer:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'price': 0.3,
            'volume': 0.3,
            'momentum': 0.2,
            'sentiment': 0.2
        }
        
    async def analyze_signals(self, signals: Dict[str, float]) -> CompositeSignal:
        """복합 신호 분석"""
        weighted_sum = sum(signals[k] * self.weights[k] 
                         for k in self.weights.keys())
        
        return CompositeSignal(
            signal_strength=abs(weighted_sum),
            signal_direction='buy' if weighted_sum > 0 else 'sell',
            confidence_score=self._calculate_confidence(signals),
            component_signals=signals
        )
