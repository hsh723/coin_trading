from typing import Dict, List
from dataclasses import dataclass

@dataclass
class HealthStatus:
    exchange_id: str
    is_healthy: bool
    latency: float
    error_rate: float
    service_status: Dict[str, bool]

class ApiHealthMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'check_interval': 60,
            'error_threshold': 0.05,
            'max_latency': 2000
        }
        self.health_history = []
        
    async def check_health(self, exchange_id: str) -> HealthStatus:
        """API 헬스 체크"""
        latency = await self._measure_latency(exchange_id)
        error_rate = self._calculate_error_rate(exchange_id)
        services = await self._check_services(exchange_id)
        
        status = HealthStatus(
            exchange_id=exchange_id,
            is_healthy=latency < self.config['max_latency'] and error_rate < self.config['error_threshold'],
            latency=latency,
            error_rate=error_rate,
            service_status=services
        )
        
        self.health_history.append(status)
        return status
