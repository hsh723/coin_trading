from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class PriceAlert:
    symbol: str
    price: float
    alert_type: str
    threshold: float
    timestamp: float

class PriceMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'check_interval': 0.1,  # 100ms
            'price_buffer': 0.002
        }
        self.alerts = []
        self.price_thresholds = {}
        
    async def monitor_prices(self, symbols: List[str]):
        """실시간 가격 모니터링"""
        while True:
            for symbol in symbols:
                current_price = await self._fetch_current_price(symbol)
                await self._check_price_alerts(symbol, current_price)
                await self._update_price_statistics(symbol, current_price)
            await asyncio.sleep(self.config['check_interval'])
