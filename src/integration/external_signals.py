from typing import Dict, List
import aiohttp
import json
from datetime import datetime

class ExternalSignalIntegrator:
    def __init__(self, config: Dict):
        self.api_endpoints = config['endpoints']
        self.credentials = config['credentials']
        self.signal_handlers = {}
        
    async def register_signal_source(self, source: str, handler: Callable):
        """신호 소스 등록"""
        self.signal_handlers[source] = handler
        
    async def fetch_signals(self) -> List[Dict]:
        """외부 신호 수집"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_source_signals(session, source, endpoint)
                for source, endpoint in self.api_endpoints.items()
            ]
            return await asyncio.gather(*tasks)
