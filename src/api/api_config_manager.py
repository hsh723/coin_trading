from typing import Dict, Optional
from dataclasses import dataclass
import yaml

@dataclass
class ApiConfig:
    exchange_id: str
    endpoints: Dict[str, str]
    rate_limits: Dict[str, int]
    timeout_settings: Dict[str, float]

class ApiConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.configs = {}
        
    async def load_api_config(self, exchange_id: str) -> ApiConfig:
        """API 설정 로드"""
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        if exchange_id not in config_data:
            raise ValueError(f"Configuration not found for exchange: {exchange_id}")
            
        return ApiConfig(
            exchange_id=exchange_id,
            endpoints=config_data[exchange_id]['endpoints'],
            rate_limits=config_data[exchange_id]['rate_limits'],
            timeout_settings=config_data[exchange_id]['timeouts']
        )
