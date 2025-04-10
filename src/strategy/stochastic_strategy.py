from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class StochasticSignal:
    signal_type: str
    k_value: float
    d_value: float
    overbought: bool
    oversold: bool

class StochasticStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'k_period': 14,
            'd_period': 3,
            'overbought': 80,
            'oversold': 20
        }
        
    async def generate_signal(self, market_data: pd.DataFrame) -> StochasticSignal:
        """스토캐스틱 신호 생성"""
        k_line, d_line = self._calculate_stochastic(market_data)
        
        return StochasticSignal(
            signal_type=self._determine_signal(k_line[-1], d_line[-1]),
            k_value=k_line[-1],
            d_value=d_line[-1],
            overbought=k_line[-1] > self.config['overbought'],
            oversold=k_line[-1] < self.config['oversold']
        )
