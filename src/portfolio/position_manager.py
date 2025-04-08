import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PositionInfo:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    risk_metrics: Dict[str, float]

class PositionManager:
    def __init__(self, risk_limits: Dict = None):
        self.risk_limits = risk_limits or {
            'max_position_size': 0.2,
            'max_drawdown': 0.1
        }
        self.positions: Dict[str, PositionInfo] = {}
        
    async def update_positions(self, market_data: Dict) -> Dict[str, PositionInfo]:
        """포지션 상태 업데이트"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                pnl = (current_price - position.entry_price) * position.size
                position.current_price = current_price
                position.pnl = pnl
                
        return self.positions
