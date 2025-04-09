from typing import Dict, List
from enum import Enum
from dataclasses import dataclass

class TradeStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

class TradeLifecycleManager:
    def __init__(self):
        self.active_trades = {}
        
    async def process_trade(self, trade: Dict) -> Dict:
        """거래 생명주기 관리"""
        trade_id = trade['id']
        self.active_trades[trade_id] = trade
        
        try:
            verified_trade = await self._verify_trade(trade)
            executed_trade = await self._execute_trade(verified_trade)
            return await self._finalize_trade(executed_trade)
        except Exception as e:
            await self._handle_trade_failure(trade_id, str(e))
            raise
