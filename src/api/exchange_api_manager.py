from typing import Dict, List
from dataclasses import dataclass
import ccxt
import asyncio

@dataclass
class ApiStatus:
    exchange_id: str
    connected: bool
    rate_limit: Dict
    permissions: List[str]
    last_request: float

class ExchangeApiManager:
    def __init__(self, api_config: Dict = None):
        self.config = api_config or {}
        self.exchanges = {}
        self.api_status = {}
        
    async def initialize_exchange(self, exchange_id: str, credentials: Dict) -> ApiStatus:
        """거래소 API 초기화"""
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': credentials['api_key'],
                'secret': credentials['secret'],
                'enableRateLimit': True
            })
            
            self.exchanges[exchange_id] = exchange
            status = await self._check_api_status(exchange_id)
            self.api_status[exchange_id] = status
            
            return status
        except Exception as e:
            await self._handle_api_error(e, exchange_id)
