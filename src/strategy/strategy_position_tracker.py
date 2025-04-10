from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class PositionStatus:
    position_id: str
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    holding_time: int

class StrategyPositionTracker:
    def __init__(self):
        self.active_positions = {}
        self.position_history = []
        
    async def track_position(self, 
                           position_id: str, 
                           market_data: Dict) -> PositionStatus:
        """포지션 상태 추적"""
        if position_id not in self.active_positions:
            return None
            
        position = self.active_positions[position_id]
        current_price = market_data['close']
        
        status = PositionStatus(
            position_id=position_id,
            symbol=position['symbol'],
            size=position['size'],
            entry_price=position['entry_price'],
            current_price=current_price,
            unrealized_pnl=self._calculate_unrealized_pnl(position, current_price),
            holding_time=self._calculate_holding_time(position)
        )
        
        return status
