from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class Watermark:
    task_id: str
    timestamp: float
    last_processed: float
    lag: float

class WatermarkManager:
    def __init__(self):
        self.watermarks = {}
        
    async def update_watermark(self, task_id: str, 
                             event_time: float) -> Watermark:
        """워터마크 업데이트"""
        current_time = time.time()
        
        watermark = Watermark(
            task_id=task_id,
            timestamp=current_time,
            last_processed=event_time,
            lag=current_time - event_time
        )
        
        self.watermarks[task_id] = watermark
        await self._check_lag_threshold(watermark)
        
        return watermark
