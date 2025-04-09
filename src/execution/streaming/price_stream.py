import asyncio
from typing import Dict, Callable
import websockets
import json

class PriceStreamHandler:
    def __init__(self):
        self.callbacks = {}
        self.connections = {}
        self.running = False
        
    async def start_streaming(self, symbols: List[str], callback: Callable):
        """가격 스트리밍 시작"""
        self.running = True
        for symbol in symbols:
            connection = await self._create_connection(symbol)
            self.connections[symbol] = connection
            asyncio.create_task(self._process_stream(symbol, connection))
            self.callbacks[symbol] = callback
