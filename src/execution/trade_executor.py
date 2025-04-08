from typing import Dict
import asyncio
import pandas as pd
from ..exchange.base import ExchangeBase

class TradeExecutor:
    def __init__(self, exchange: ExchangeBase, config: Dict):
        self.exchange = exchange
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
    async def execute_trade(self, order: Dict) -> Dict:
        """주문 실행"""
        for attempt in range(self.max_retries):
            try:
                result = await self._place_order(order)
                if result['status'] == 'filled':
                    return result
                    
                await self._monitor_order(result['order_id'])
                return await self._get_order_result(result['order_id'])
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(self.retry_delay)
