from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Position:
    symbol: str
    entry_price: float
    size: float
    side: str
    pnl: float = 0.0

class BacktestPositionTracker:
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict] = []
        
    def update_positions(self, market_data: Dict) -> None:
        """포지션 업데이트"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['close']
                position.pnl = self._calculate_pnl(position, current_price)
