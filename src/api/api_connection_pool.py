from typing import Dict, Optional
from dataclasses import dataclass
import aiohttp
import asyncio

@dataclass
class ConnectionInfo:
    pool_id: str
    active_connections: int
    idle_connections: int
    max_connections: int

class ApiConnectionPool:
    def __init__(self, pool_config: Dict = None):
        self.config = pool_config or {
            'max_connections': 10,
            'min_connections': 2,
            'connection_timeout': 30
        }
        self.pools = {}
        
    async def get_connection(self, exchange_id: str) -> aiohttp.ClientSession:
        """API 연결 풀에서 연결 가져오기"""
        if exchange_id not in self.pools:
            await self._create_pool(exchange_id)
            
        pool = self.pools[exchange_id]
        return await self._acquire_connection(pool)
