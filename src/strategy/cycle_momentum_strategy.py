from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class CycleMomentumSignal:
    cycle_position: float  # 0 to 1
    momentum_score: float
    trend_alignment: bool
    entry_points: List[float]

class CycleMomentumStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'cycle_period': 20,
            'momentum_period': 10,
            'signal_threshold': 0.7
        }
        
    async def generate_signals(self, market_data: pd.DataFrame) -> CycleMomentumSignal:
        """사이클 모멘텀 신호 생성"""
        cycle_pos = self._calculate_cycle_position(market_data)
        momentum = self._calculate_momentum(market_data)
        trend = self._analyze_trend(market_data)
        
        return CycleMomentumSignal(
            cycle_position=cycle_pos,
            momentum_score=momentum,
            trend_alignment=self._check_trend_alignment(cycle_pos, trend),
            entry_points=self._find_entry_points(cycle_pos, momentum)
        )
