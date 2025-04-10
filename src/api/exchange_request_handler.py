from typing import Dict, Any
from dataclasses import dataclass
import asyncio
import time

@dataclass
class RequestMetrics:
    latency: float
    success: bool
    error_type: str = None
    rate_limit_remaining: int = None

class ExchangeRequestHandler:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_retries': 3,
            'retry_delay': 1.0,
            'timeout': 30
        }
        self.request_history = []
        
    async def execute_request(self, 
                            exchange: Any, 
                            method: str, 
                            **params) -> Dict:
        """거래소 API 요청 실행"""
        for attempt in range(self.config['max_retries']):
            try:
                start_time = time.time()
                response = await getattr(exchange, method)(**params)
                latency = time.time() - start_time
                
                metrics = RequestMetrics(
                    latency=latency,
                    success=True,
                    rate_limit_remaining=exchange.rateLimit
                )
                self.request_history.append(metrics)
                
                return response
            except Exception as e:
                await self._handle_request_error(e, attempt)
