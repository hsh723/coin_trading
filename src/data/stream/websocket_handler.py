import asyncio
import websockets
from typing import Dict, Callable
import json

class WebsocketHandler:
    def __init__(self):
        self.connections = {}
        self.callbacks = {}
        self.running = False
        
    async def connect_and_subscribe(self, url: str, 
                                  symbol: str, 
                                  callback: Callable):
        """웹소켓 연결 및 구독"""
        try:
            connection = await websockets.connect(url)
            self.connections[symbol] = connection
            self.callbacks[symbol] = callback
            
            await self._subscribe(connection, symbol)
            asyncio.create_task(self._message_handler(symbol, connection))
        except Exception as e:
            await self._handle_connection_error(symbol, e)
