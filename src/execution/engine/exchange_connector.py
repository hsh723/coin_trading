from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class ExchangeStatus:
    connected: bool
    latency: float
    rate_limits: Dict[str, int]
    current_load: float

class ExchangeConnector:
    def __init__(self, exchange_config: Dict = None):
        self.config = exchange_config or {
            'retry_attempts': 3,
            'timeout': 10,
            'heartbeat_interval': 30
        }
        self.connections = {}
        
    async def connect_exchange(self, exchange_id: str) -> ExchangeStatus:
        """거래소 연결 관리"""
        try:
            connection = await self._establish_connection(exchange_id)
            status = await self._check_exchange_status(connection)
            
            self.connections[exchange_id] = {
                'connection': connection,
                'status': status,
                'last_heartbeat': time.time()
            }
            
            asyncio.create_task(self._maintain_connection(exchange_id))
            return status
            
        except Exception as e:
            await self._handle_connection_error(e, exchange_id)
            raise
