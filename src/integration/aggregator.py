import pandas as pd
from typing import Dict, List
import asyncio

class MarketDataAggregator:
    def __init__(self, exchanges: List[str]):
        self.exchanges = exchanges
        self.data_cache = {}
        
    async def aggregate_market_data(self, symbol: str) -> pd.DataFrame:
        """여러 거래소의 시장 데이터 집계"""
        tasks = [
            self._fetch_exchange_data(exchange, symbol)
            for exchange in self.exchanges
        ]
        results = await asyncio.gather(*tasks)
        return self._combine_exchange_data(results)
