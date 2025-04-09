from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class TradeStatus:
    order_id: str
    status: str
    execution_progress: float
    average_price: float
    filled_amount: float
    remaining_amount: float

class RealTimeTradeMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'update_interval': 1.0,
            'max_monitoring_time': 300
        }
        self.active_trades = {}
        
    async def monitor_trade(self, trade_id: str) -> TradeStatus:
        """실시간 거래 모니터링"""
        start_time = time.time()
        while time.time() - start_time < self.config['max_monitoring_time']:
            status = await self._fetch_trade_status(trade_id)
            if status.status in ['filled', 'cancelled', 'failed']:
                return status
            await asyncio.sleep(self.config['update_interval'])
