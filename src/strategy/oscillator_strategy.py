from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class OscillatorSignal:
    signal_type: str
    strength: float
    overbought: bool
    oversold: bool
    divergence: Dict[str, bool]

class OscillatorStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'rsi_period': 14,
            'stoch_period': 14,
            'cci_period': 20,
            'overbought_threshold': 70,
            'oversold_threshold': 30
        }
        
    async def generate_signals(self, market_data: pd.DataFrame) -> OscillatorSignal:
        """오실레이터 신호 생성"""
        rsi = self._calculate_rsi(market_data['close'])
        stoch = self._calculate_stochastic(market_data)
        cci = self._calculate_cci(market_data)
        
        return OscillatorSignal(
            signal_type=self._determine_signal(rsi, stoch, cci),
            strength=self._calculate_signal_strength(rsi, stoch, cci),
            overbought=self._check_overbought(rsi, stoch),
            oversold=self._check_oversold(rsi, stoch),
            divergence=self._check_divergence(market_data, rsi)
        )
