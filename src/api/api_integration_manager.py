from typing import Dict, List
from dataclasses import dataclass

@dataclass
class IntegrationStatus:
    status: str
    connected_apis: List[str]
    health_check: Dict[str, bool]
    error_count: Dict[str, int]

class ApiIntegrationManager:
    def __init__(self, integration_config: Dict = None):
        self.config = integration_config or {
            'health_check_interval': 300,
            'auto_reconnect': True
        }
        self.integrations = {}
        
    async def initialize_integrations(self) -> IntegrationStatus:
        """API 통합 초기화"""
        status = IntegrationStatus(
            status='initializing',
            connected_apis=[],
            health_check={},
            error_count={}
        )
        
        try:
            for api_name, api_config in self.config.items():
                await self._initialize_single_api(api_name, api_config)
                status.connected_apis.append(api_name)
                status.health_check[api_name] = True
                status.error_count[api_name] = 0
                
            status.status = 'running'
        except Exception as e:
            status.status = 'error'
            await self._handle_initialization_error(e)
            
        return status
