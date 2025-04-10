from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    liquidation_price: float
    margin_ratio: float

class PositionManager:
    def __init__(self, risk_limits: Dict = None):
        self.risk_limits = risk_limits or {
            'max_position_size': 1.0,
            'max_leverage': 10.0,
            'min_margin_ratio': 0.05
        }
        self.positions = {}
        
    async def update_position(self, symbol: str, 
                            execution_data: Dict) -> Position:
        """포지션 업데이트"""
        if symbol not in self.positions:
            self.positions[symbol] = self._create_position(execution_data)
        else:
            self._modify_position(symbol, execution_data)
            
        return await self._calculate_position_metrics(symbol)
