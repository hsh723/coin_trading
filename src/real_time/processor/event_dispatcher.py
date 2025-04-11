import asyncio
from typing import Dict, Callable
from dataclasses import dataclass

@dataclass
class EventMetrics:
    processed_count: int
    error_count: int
    avg_processing_time: float
    queue_size: int

class EventDispatcher:
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self.metrics = EventMetrics(0, 0, 0.0, 0)
        
    async def dispatch_event(self, event_type: str, event_data: Dict) -> bool:
        """이벤트 분배 처리"""
        if event_type in self.handlers:
            start_time = asyncio.get_event_loop().time()
            
            try:
                for handler in self.handlers[event_type]:
                    await handler(event_data)
                self.metrics.processed_count += 1
                
            except Exception as e:
                self.metrics.error_count += 1
                return False
                
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(processing_time)
            
        return True
