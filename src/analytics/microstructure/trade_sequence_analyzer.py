import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class TradeSequence:
    sequence_pattern: str
    trade_intensity: float
    sequence_momentum: float
    trade_clustering: Dict[str, float]

class TradeSequenceAnalyzer:
    def __init__(self, sequence_window: int = 100):
        self.sequence_window = sequence_window
        
    async def analyze_sequence(self, trades: List[Dict]) -> TradeSequence:
        """거래 시퀀스 분석"""
        recent_trades = trades[-self.sequence_window:]
        pattern = self._identify_sequence_pattern(recent_trades)
        
        return TradeSequence(
            sequence_pattern=pattern,
            trade_intensity=self._calculate_intensity(recent_trades),
            sequence_momentum=self._calculate_momentum(recent_trades),
            trade_clustering=self._analyze_clustering(recent_trades)
        )
