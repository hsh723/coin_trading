from typing import Dict, Optional
from dataclasses import dataclass
import aiohttp
import time

@dataclass
class ClientStatus:
    client_id: str
    active: bool
    last_request: float
    request_count: int
    error_count: int

class ApiClientManager:
    def __init__(self, client_config: Dict = None):
        self.config = client_config or {
            'max_clients': 5,
            'request_timeout': 30,
            'client_ttl': 3600  # 1시간
        }
        self.clients = {}
        
    async def get_client(self, exchange_id: str) -> Optional[aiohttp.ClientSession]:
        """API 클라이언트 가져오기"""
        if exchange_id in self.clients:
            return await self._validate_and_return_client(exchange_id)
            
        return await self._create_new_client(exchange_id)
