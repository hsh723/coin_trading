import aiohttp
import pandas as pd
from typing import Dict, List
import asyncio

class AlternativeDataCollector:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        
    async def collect_funding_rates(self, exchanges: List[str]) -> pd.DataFrame:
        """펀딩비율 데이터 수집"""
        tasks = [self._fetch_funding_rate(exchange) for exchange in exchanges]
        results = await asyncio.gather(*tasks)
        return pd.concat(results)

    async def collect_liquidations(self, timeframe: str) -> pd.DataFrame:
        """청산 데이터 수집"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"api/liquidations/{timeframe}") as response:
                data = await response.json()
                return pd.DataFrame(data)
