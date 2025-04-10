from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class QuotaInfo:
    total_quota: int
    remaining_quota: int
    reset_time: float
    quota_type: str

class ApiQuotaManager:
    def __init__(self, quota_config: Dict = None):
        self.config = quota_config or {
            'default_quota': 1000,
            'quota_window': 3600,  # 1시간
            'warning_threshold': 0.2
        }
        self.quotas = {}
        
    async def check_quota(self, exchange_id: str) -> Optional[QuotaInfo]:
        """API 할당량 체크"""
        if exchange_id not in self.quotas:
            return self._initialize_quota(exchange_id)
            
        quota = self.quotas[exchange_id]
        if self._should_reset_quota(quota):
            return self._reset_quota(exchange_id)
            
        return quota
