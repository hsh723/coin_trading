from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class ROCSignal:
    signal_type: str
    roc_value: float
    momentum_strength: float
    trend_reversal: bool
    threshold_breach: bool

class ROCStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'roc_period': 12,
            'signal_threshold': 0.02,
            'momentum_lookback': 3
        }
        
    async def generate_signal(self, market_data: pd.DataFrame) -> ROCSignal:
        """ROC 신호 생성"""
        roc = self._calculate_roc(market_data['close'])
        momentum = self._calculate_momentum_strength(roc)
        
        return ROCSignal(
            signal_type=self._determine_signal(roc[-1]),
            roc_value=roc[-1],
            momentum_strength=momentum,
            trend_reversal=self._detect_reversal(roc),
            threshold_breach=abs(roc[-1]) > self.config['signal_threshold']
        )
