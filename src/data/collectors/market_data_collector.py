from typing import Dict, List
import asyncio
import aiohttp
from datetime import datetime

class MarketDataCollector:
    def __init__(self, exchanges: List[str], config: Dict = None):
        self.exchanges = exchanges
        self.config = config or {
            'request_timeout': 30,
            'retry_attempts': 3
        }
        
    async def collect_market_data(self, symbols: List[str]) -> Dict:
        """시장 데이터 수집"""
        tasks = []
        for exchange in self.exchanges:
            for symbol in symbols:
                tasks.append(self._fetch_data(exchange, symbol))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._process_results(results, symbols)
