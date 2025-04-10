from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class SwingPoint:
    price: float
    timestamp: float
    type: str  # 'high' or 'low'
    strength: float

class SwingTradeStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'swing_threshold': 0.02,  # 2%
            'min_swing_periods': 3,
            'profit_target': 0.03,    # 3%
            'stop_loss': 0.015        # 1.5%
        }
        
    async def analyze_swings(self, price_data: pd.DataFrame) -> List[SwingPoint]:
        """스윙 포인트 분석"""
        highs = self._find_swing_highs(price_data)
        lows = self._find_swing_lows(price_data)
        
        swing_points = []
        for point in sorted(highs + lows, key=lambda x: x.timestamp):
            if self._validate_swing_point(point, swing_points):
                swing_points.append(point)
                
        return swing_points
