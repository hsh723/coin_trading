from typing import Dict, List
import pandas as pd
from .base import BaseStrategy

class HybridStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], weights: List[float]):
        super().__init__()
        self.strategies = strategies
        self.weights = weights
        assert len(strategies) == len(weights), "전략과 가중치 수가 일치해야 합니다"
        assert abs(sum(weights) - 1.0) < 1e-6, "가중치 합은 1이어야 합니다"

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """여러 전략의 신호를 결합"""
        signals = []
        for strategy, weight in zip(self.strategies, self.weights):
            strategy_signal = strategy.generate_signals(data)
            signals.append((strategy_signal, weight))
            
        return self._combine_signals(signals)
