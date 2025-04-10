from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class RSISignal:
    signal_type: str
    rsi_value: float
    overbought: bool
    oversold: bool
    divergence_detected: bool

class RSIStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'rsi_period': 14,
            'overbought_level': 70,
            'oversold_level': 30,
            'divergence_lookback': 14
        }
        
    async def generate_signal(self, market_data: pd.DataFrame) -> RSISignal:
        """RSI 신호 생성"""
        rsi = self._calculate_rsi(market_data['close'])
        current_rsi = rsi[-1]
        
        return RSISignal(
            signal_type=self._determine_signal(current_rsi),
            rsi_value=current_rsi,
            overbought=current_rsi > self.config['overbought_level'],
            oversold=current_rsi < self.config['oversold_level'],
            divergence_detected=self._check_divergence(market_data['close'], rsi)
        )
