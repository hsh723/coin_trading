from fastapi import FastAPI, Request, HTTPException
from typing import Dict, Callable
import hmac
import hashlib

class WebhookHandler:
    def __init__(self, secret_key: str):
        self.app = FastAPI()
        self.secret_key = secret_key
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """이벤트 핸들러 등록"""
        self.handlers[event_type] = handler

    async def verify_signature(self, payload: bytes, signature: str) -> bool:
        """웹훅 서명 검증"""
        computed = hmac.new(
            self.secret_key.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed, signature)
