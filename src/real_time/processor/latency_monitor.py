import asyncio
from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class LatencyMetrics:
    processing_latency: float
    network_latency: float
    execution_latency: float
    total_latency: float

class LatencyMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'sampling_interval': 0.1,
            'history_size': 1000
        }
        self.latency_history = []
        
    async def monitor_latency(self, execution_data: Dict) -> LatencyMetrics:
        """실시간 지연시간 모니터링"""
        processing_time = self._measure_processing_time(execution_data)
        network_time = self._measure_network_time(execution_data)
        execution_time = self._measure_execution_time(execution_data)
        
        metrics = LatencyMetrics(
            processing_latency=processing_time,
            network_latency=network_time,
            execution_latency=execution_time,
            total_latency=processing_time + network_time + execution_time
        )
        
        await self._update_history(metrics)
        return metrics
