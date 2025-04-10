from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ApiVersion:
    version: str
    supported: bool
    deprecated: bool
    sunset_date: Optional[str]
    features: List[str]

class ApiVersionManager:
    def __init__(self, version_config: Dict = None):
        self.config = version_config or {
            'default_version': 'v1',
            'supported_versions': ['v1', 'v2'],
            'deprecation_window': 90  # days
        }
        self.versions = {}
        
    async def check_version_compatibility(self, 
                                       exchange_id: str, 
                                       version: str) -> ApiVersion:
        """API 버전 호환성 확인"""
        if version not in self.versions:
            await self._load_version_info(version)
            
        return self.versions.get(version, self._get_default_version())
