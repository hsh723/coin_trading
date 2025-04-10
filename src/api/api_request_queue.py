from typing import Dict, List
from dataclasses import dataclass
import asyncio
import heapq

@dataclass
class QueuedRequest:
    priority: int
    timestamp: float
    request_type: str
    params: Dict
    callback: callable

class ApiRequestQueue:
    def __init__(self, queue_config: Dict = None):
        self.config = queue_config or {
            'max_queue_size': 1000,
            'priority_levels': 3
        }
        self.request_queue = []
        
    async def enqueue_request(self, request: QueuedRequest) -> bool:
        """API 요청 큐잉"""
        if len(self.request_queue) >= self.config['max_queue_size']:
            return False
            
        heapq.heappush(
            self.request_queue,
            (request.priority, request.timestamp, request)
        )
        return True
