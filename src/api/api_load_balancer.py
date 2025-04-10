from typing import Dict, List
from dataclasses import dataclass

@dataclass
class LoadBalancerMetrics:
    total_requests: int
    requests_per_endpoint: Dict[str, int]
    average_response_time: float
    endpoint_health: Dict[str, bool]

class ApiLoadBalancer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'strategy': 'round_robin',
            'health_check_interval': 60,
            'timeout': 30
        }
        self.endpoints = {}
        self.current_index = 0
        
    async def get_next_endpoint(self) -> str:
        """다음 엔드포인트 선택"""
        available_endpoints = [
            endpoint for endpoint, status in self.endpoints.items() 
            if status['healthy']
        ]
        
        if not available_endpoints:
            raise RuntimeError("No healthy endpoints available")
            
        self.current_index = (self.current_index + 1) % len(available_endpoints)
        return available_endpoints[self.current_index]
