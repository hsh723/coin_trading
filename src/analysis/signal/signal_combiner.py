from typing import Dict, List
import numpy as np

class SignalCombiner:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {}
        self.signals_history = []
        
    def combine_signals(self, signals: Dict[str, Dict]) -> Dict:
        """여러 전략의 신호 결합"""
        combined_score = 0.0
        total_weight = 0.0
        
        for strategy_name, signal in signals.items():
            weight = self.weights.get(strategy_name, 1.0)
            score = self._convert_signal_to_score(signal)
            combined_score += score * weight
            total_weight += weight
            
        combined_score /= total_weight
        return self._score_to_signal(combined_score)
