import asyncio
from typing import Dict, Callable
import websockets

class ExecutionStreamHandler:
    def __init__(self):
        self.handlers = {}
        self.active_streams = {}
        
    async def start_execution_stream(self, symbol: str, callback: Callable):
        """실행 스트림 시작"""
        if symbol not in self.active_streams:
            stream = await self._create_execution_stream(symbol)
            self.active_streams[symbol] = stream
            self.handlers[symbol] = callback
            asyncio.create_task(self._process_execution_updates(symbol))
