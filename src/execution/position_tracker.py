from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float

class PositionTracker:
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict] = []
        
    async def update_positions(self, market_data: Dict[str, float]):
        """포지션 업데이트"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                unrealized_pnl = (current_price - position.entry_price) * position.size
                position.current_price = current_price
                position.unrealized_pnl = unrealized_pnl
