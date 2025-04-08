from web3 import Web3
import pandas as pd
from typing import Dict, List
import asyncio

class OnChainDataCollector:
    def __init__(self, web3_url: str):
        self.w3 = Web3(Web3.HTTPProvider(web3_url))
        self.metrics = {}

    async def collect_metrics(self, address: str) -> Dict:
        """온체인 메트릭스 수집"""
        tasks = [
            self._get_transaction_count(address),
            self._get_balance_history(address),
            self._get_contract_interactions(address)
        ]
        results = await asyncio.gather(*tasks)
        return self._process_metrics(results)

    async def analyze_token_flows(self, token_address: str) -> pd.DataFrame:
        """토큰 흐름 분석"""
        transfers = await self._get_transfer_events(token_address)
        return self._analyze_transfers(transfers)
