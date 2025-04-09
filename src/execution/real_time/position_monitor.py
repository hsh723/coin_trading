from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class PositionState:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin_ratio: float

class PositionMonitor:
    def __init__(self, risk_limits: Dict = None):
        self.risk_limits = risk_limits or {
            'max_drawdown': 0.1,
            'margin_call_ratio': 0.8
        }
        self.active_positions = {}
        
    async def monitor_positions(self):
        """실시간 포지션 모니터링"""
        while True:
            for symbol, position in self.active_positions.items():
                state = await self._update_position_state(symbol)
                await self._check_risk_limits(state)
                await self._notify_if_needed(state)
            await asyncio.sleep(1)
