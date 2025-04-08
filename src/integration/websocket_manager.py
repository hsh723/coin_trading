import asyncio
import websockets
from typing import Dict, Callable
import json
import logging

class WebsocketManager:
    def __init__(self):
        self.connections = {}
        self.callbacks = {}
        self.logger = logging.getLogger(__name__)
        
    async def connect(self, url: str, symbol: str):
        """웹소켷 연결 및 구독"""
        try:
            connection = await websockets.connect(url)
            self.connections[symbol] = connection
            await self._subscribe(connection, symbol)
            await self._listen(connection, symbol)
        except Exception as e:
            self.logger.error(f"WebSocket error for {symbol}: {str(e)}")
