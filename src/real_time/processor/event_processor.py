import asyncio
from typing import Dict
from dataclasses import dataclass

@dataclass
class EventProcessingMetrics:
    processed_count: int
    processing_time: float
    error_rate: float
    success_rate: float

class EventProcessor:
    def __init__(self, processing_config: Dict = None):
        self.config = processing_config or {
            'batch_size': 100,
            'timeout': 5.0
        }
        
    async def process_events(self, events: List[Dict]) -> EventProcessingMetrics:
        """이벤트 처리"""
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[
            self._process_single_event(event) 
            for event in events
        ], return_exceptions=True)
        
        return EventProcessingMetrics(
            processed_count=len(results),
            processing_time=asyncio.get_event_loop().time() - start_time,
            error_rate=self._calculate_error_rate(results),
            success_rate=self._calculate_success_rate(results)
        )
