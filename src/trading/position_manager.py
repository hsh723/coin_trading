from typing import Dict, List
import pandas as pd
from ..risk.manager import RiskManager

class PositionManager:
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        
    def open_position(self, symbol: str, side: str, entry_price: float, amount: float) -> Dict:
        """새로운 포지션 오픈"""
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'amount': amount,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'stop_loss': self.risk_manager.calculate_stop_loss(entry_price, side),
            'take_profit': self.risk_manager.calculate_take_profit(entry_price, side)
        }
        
        self.positions[symbol] = position
        return position
        
    def update_position(self, symbol: str, current_price: float) -> None:
        """포지션 업데이트"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position['current_price'] = current_price
            position['unrealized_pnl'] = self._calculate_pnl(position)
            
    def close_position(self, symbol: str, exit_price: float) -> Dict:
        """포지션 종료"""
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            position['exit_price'] = exit_price
            position['realized_pnl'] = self._calculate_pnl(position)
            self.position_history.append(position)
            return position
        return None
