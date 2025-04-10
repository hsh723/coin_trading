from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class MACDSignal:
    signal_type: str
    macd_line: float
    signal_line: float
    histogram: float
    crossover_detected: bool

class MACDStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
        
    async def generate_signal(self, market_data: pd.DataFrame) -> MACDSignal:
        """MACD 신호 생성"""
        macd_data = self._calculate_macd(market_data['close'])
        
        return MACDSignal(
            signal_type=self._determine_signal(macd_data),
            macd_line=macd_data['macd'][-1],
            signal_line=macd_data['signal'][-1],
            histogram=macd_data['hist'][-1],
            crossover_detected=self._detect_crossover(macd_data)
        )
