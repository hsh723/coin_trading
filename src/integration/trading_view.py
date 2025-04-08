from fastapi import FastAPI, Request
from typing import Dict
import json
import asyncio

class TradingViewIntegration:
    def __init__(self, webhook_secret: str):
        self.app = FastAPI()
        self.webhook_secret = webhook_secret
        self.signal_queue = asyncio.Queue()

        @self.app.post("/webhook")
        async def receive_alert(request: Request):
            data = await request.json()
            if self._verify_webhook(data):
                await self._process_alert(data)
                return {"status": "success"}
            return {"status": "error", "message": "Invalid webhook"}

    async def _process_alert(self, data: Dict) -> None:
        """TradingView 알림 처리"""
        signal = self._parse_alert_data(data)
        await self.signal_queue.put(signal)
