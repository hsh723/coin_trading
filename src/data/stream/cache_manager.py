from typing import Dict, Any
import time
from collections import OrderedDict

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        
    async def get_or_fetch(self, key: str, fetch_func) -> Any:
        """캐시된 데이터 조회 또는 가져오기"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
                
        value = await fetch_func()
        self._update_cache(key, value)
        return value
