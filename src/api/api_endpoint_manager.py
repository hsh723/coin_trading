from typing import Dict, List
from dataclasses import dataclass

@dataclass
class EndpointStatus:
    endpoint_id: str
    url: str
    method: str
    rate_limit: Dict
    availability: float

class ApiEndpointManager:
    def __init__(self, endpoint_config: Dict = None):
        self.config = endpoint_config or {
            'health_check_interval': 60,
            'retry_attempts': 3
        }
        self.endpoints = {}
        
    async def register_endpoint(self, endpoint_id: str, 
                              endpoint_data: Dict) -> EndpointStatus:
        """API 엔드포인트 등록"""
        status = EndpointStatus(
            endpoint_id=endpoint_id,
            url=endpoint_data['url'],
            method=endpoint_data['method'],
            rate_limit=endpoint_data.get('rate_limit', {}),
            availability=1.0
        )
        
        self.endpoints[endpoint_id] = {
            'data': endpoint_data,
            'status': status
        }
        
        return status
