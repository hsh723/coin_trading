import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CombinedSignal:
    direction: str
    strength: float
    confidence: float
    sources: List[str]

class SignalCombiner:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {}
        self.signal_history = []
        
    def combine_signals(self, signals: Dict[str, Dict]) -> CombinedSignal:
        """여러 소스의 신호 결합"""
        weighted_sum = 0
        total_weight = 0
        active_sources = []
        
        for source, signal in signals.items():
            weight = self.weights.get(source, 1.0)
            score = self._convert_signal_to_score(signal)
            weighted_sum += score * weight
            total_weight += weight
            
            if abs(score) > 0.1:  # 유의미한 신호만 기록
                active_sources.append(source)
                
        combined_score = weighted_sum / total_weight if total_weight > 0 else 0
        return self._create_combined_signal(combined_score, active_sources)
