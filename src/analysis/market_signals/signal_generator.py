import pandas as pd
from typing import Dict, List
from ...strategy.base import BaseStrategy

class MarketSignalGenerator:
    def __init__(self, strategies: List[BaseStrategy]):
        self.strategies = strategies
        self.signal_weights = {}
        
    async def generate_combined_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """여러 전략의 신호를 결합"""
        signals = {}
        for strategy in self.strategies:
            strategy_signal = await strategy.generate_signals(market_data)
            self._update_signal_weights(strategy, strategy_signal)
            signals[strategy.__class__.__name__] = strategy_signal
            
        return self._combine_weighted_signals(signals)
