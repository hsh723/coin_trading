from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

@dataclass
class CacheEntry:
    key: str
    data: Any
    timestamp: float
    ttl: int
    hits: int

class ApiCacheManager:
    def __init__(self, cache_config: Dict = None):
        self.config = cache_config or {
            'default_ttl': 300,  # 5분
            'max_size': 1000,
            'cleanup_interval': 3600
        }
        self.cache = {}
        
    async def get_or_fetch(self, 
                          key: str, 
                          fetch_func: callable,
                          ttl: Optional[int] = None) -> Any:
        """캐시된 데이터 조회 또는 가져오기"""
        if self._is_valid_cache(key):
            self.cache[key].hits += 1
            return self.cache[key].data
            
        data = await fetch_func()
        self._store_cache(key, data, ttl)
        return data
