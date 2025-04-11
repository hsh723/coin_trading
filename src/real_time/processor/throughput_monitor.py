import asyncio
from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class ThroughputMetrics:
    orders_per_second: float
    messages_per_second: float
    processing_capacity: float
    system_load: float

class ThroughputMonitor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.metrics_buffer = []
        
    async def monitor_throughput(self) -> ThroughputMetrics:
        """실시간 처리량 모니터링"""
        current_metrics = await self._collect_current_metrics()
        self.metrics_buffer.append(current_metrics)
        
        if len(self.metrics_buffer) > self.window_size:
            self.metrics_buffer.pop(0)
            
        return ThroughputMetrics(
            orders_per_second=self._calculate_orders_per_second(),
            messages_per_second=self._calculate_messages_per_second(),
            processing_capacity=self._calculate_processing_capacity(),
            system_load=self._calculate_system_load()
        )
