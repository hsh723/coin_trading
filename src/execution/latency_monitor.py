from typing import Dict, List
from dataclasses import dataclass
import time
import numpy as np

@dataclass
class LatencyMetrics:
    average_latency: float
    max_latency: float
    min_latency: float
    latency_percentiles: Dict[str, float]

class LatencyMonitor:
    def __init__(self):
        self.latency_history = []
        self.threshold = 500  # 밀리초
        
    async def measure_latency(self, operation: str) -> float:
        """작업 지연 시간 측정"""
        start_time = time.time()
        try:
            result = await self._execute_operation(operation)
            latency = (time.time() - start_time) * 1000  # 밀리초로 변환
            self.latency_history.append(latency)
            return result
        except Exception as e:
            self._handle_latency_error(operation, e)
            raise
