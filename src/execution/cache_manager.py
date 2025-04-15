"""
실행 시스템 캐시 관리자
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger(__name__)

class CacheEntry:
    """캐시 항목"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,  # 기본 TTL: 1시간
        max_size: int = 1024 * 1024  # 기본 최대 크기: 1MB
    ):
        """
        캐시 항목 초기화
        
        Args:
            key (str): 캐시 키
            value (Any): 캐시 값
            ttl (int): Time-To-Live (초)
            max_size (int): 최대 크기 (바이트)
        """
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=ttl)
        self.last_accessed = self.created_at
        self.access_count = 0
        self.size = len(json.dumps(value)) if value is not None else 0
        self.max_size = max_size
        
    def is_expired(self) -> bool:
        """
        만료 여부 확인
        
        Returns:
            bool: 만료 여부
        """
        return datetime.now() > self.expires_at
        
    def is_oversized(self) -> bool:
        """
        크기 초과 여부 확인
        
        Returns:
            bool: 크기 초과 여부
        """
        return self.size > self.max_size
        
    def access(self) -> None:
        """캐시 접근 기록"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
class CacheManager:
    """실행 시스템 캐시 관리자"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        캐시 관리자 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        
        # 캐시 설정
        cache_config = config.get('cache', {})
        self.max_entries = cache_config.get('max_entries', 1000)
        self.max_size = cache_config.get('max_size', 100 * 1024 * 1024)  # 100MB
        self.default_ttl = cache_config.get('default_ttl', 3600)  # 1시간
        
        # 캐시 저장소
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size = 0
        
        # 통계
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0
        }
        
        # 청소 작업 스케줄링
        self.cleanup_interval = cache_config.get('cleanup_interval', 300)  # 5분
        self.cleanup_task = None
        
    async def initialize(self):
        """캐시 관리자 초기화"""
        # 주기적 청소 작업 시작
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("캐시 관리자 초기화 완료")
        
    async def close(self):
        """리소스 정리"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
                
        self.clear()
        logger.info("캐시 관리자 종료")
        
    def cache_execution(self, ttl: Optional[int] = None):
        """
        실행 결과 캐싱 데코레이터
        
        Args:
            ttl (Optional[int]): Time-To-Live (초)
            
        Returns:
            실행 결과를 캐싱하는 데코레이터
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = self._generate_cache_key(func, args, kwargs)
                
                # 캐시 확인
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                    
                # 함수 실행
                result = await func(*args, **kwargs)
                
                # 결과 캐싱
                await self.set(
                    cache_key,
                    result,
                    ttl or self.default_ttl
                )
                
                return result
                
            return wrapper
            
        return decorator
        
    async def get(self, key: str) -> Optional[Any]:
        """
        캐시 조회
        
        Args:
            key (str): 캐시 키
            
        Returns:
            Optional[Any]: 캐시된 값
        """
        entry = self.cache.get(key)
        
        if entry is None:
            self.stats['misses'] += 1
            return None
            
        if entry.is_expired():
            await self.delete(key)
            self.stats['expirations'] += 1
            return None
            
        # 접근 기록 갱신
        entry.access()
        self.stats['hits'] += 1
        
        # LRU 순서 갱신
        self.cache.move_to_end(key)
        
        return entry.value
        
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        캐시 설정
        
        Args:
            key (str): 캐시 키
            value (Any): 캐시할 값
            ttl (Optional[int]): Time-To-Live (초)
            
        Returns:
            bool: 성공 여부
        """
        # 기존 항목 삭제
        if key in self.cache:
            await self.delete(key)
            
        # 새 항목 생성
        entry = CacheEntry(
            key,
            value,
            ttl or self.default_ttl
        )
        
        # 크기 검증
        if entry.is_oversized():
            logger.warning(f"캐시 항목 크기 초과: {key} ({entry.size} bytes)")
            return False
            
        # 공간 확보
        while (
            len(self.cache) >= self.max_entries or
            self.total_size + entry.size > self.max_size
        ):
            await self._evict_entry()
            
        # 항목 추가
        self.cache[key] = entry
        self.total_size += entry.size
        self.cache.move_to_end(key)
        
        return True
        
    async def delete(self, key: str) -> bool:
        """
        캐시 삭제
        
        Args:
            key (str): 캐시 키
            
        Returns:
            bool: 성공 여부
        """
        entry = self.cache.pop(key, None)
        if entry:
            self.total_size -= entry.size
            return True
        return False
        
    def clear(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        self.total_size = 0
        
    async def _periodic_cleanup(self):
        """주기적 캐시 청소"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"캐시 청소 중 오류 발생: {str(e)}")
                
    async def _cleanup_expired(self):
        """만료된 항목 정리"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            await self.delete(key)
            self.stats['expirations'] += 1
            
    async def _evict_entry(self) -> bool:
        """
        캐시 항목 제거
        
        Returns:
            bool: 성공 여부
        """
        if not self.cache:
            return False
            
        # LRU 정책에 따라 가장 오래된 항목 제거
        key, entry = self.cache.popitem(last=False)
        self.total_size -= entry.size
        self.stats['evictions'] += 1
        
        return True
        
    def _generate_cache_key(
        self,
        func,
        args: Tuple,
        kwargs: Dict
    ) -> str:
        """
        캐시 키 생성
        
        Args:
            func: 대상 함수
            args (Tuple): 위치 인자
            kwargs (Dict): 키워드 인자
            
        Returns:
            str: 캐시 키
        """
        # 함수 정보
        func_info = f"{func.__module__}.{func.__name__}"
        
        # 인자 정보
        args_info = [str(arg) for arg in args]
        kwargs_info = [f"{k}={v}" for k, v in sorted(kwargs.items())]
        
        # 키 생성
        key_parts = [func_info] + args_info + kwargs_info
        return ":".join(key_parts)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 조회
        
        Returns:
            Dict[str, Any]: 캐시 통계
        """
        stats = self.stats.copy()
        stats.update({
            'entries': len(self.cache),
            'total_size': self.total_size,
            'hit_ratio': (
                stats['hits'] / (stats['hits'] + stats['misses'])
                if stats['hits'] + stats['misses'] > 0
                else 0
            )
        })
        return stats
        
    async def get_entries_by_pattern(
        self,
        pattern: str
    ) -> List[Tuple[str, Any]]:
        """
        패턴으로 캐시 항목 조회
        
        Args:
            pattern (str): 검색 패턴
            
        Returns:
            List[Tuple[str, Any]]: 캐시 항목 목록
        """
        import re
        regex = re.compile(pattern)
        
        entries = []
        for key, entry in self.cache.items():
            if regex.match(key):
                if not entry.is_expired():
                    entries.append((key, entry.value))
                    
        return entries
        
    async def update_ttl(
        self,
        key: str,
        ttl: int
    ) -> bool:
        """
        TTL 업데이트
        
        Args:
            key (str): 캐시 키
            ttl (int): 새로운 TTL (초)
            
        Returns:
            bool: 성공 여부
        """
        entry = self.cache.get(key)
        if entry and not entry.is_expired():
            entry.expires_at = datetime.now() + timedelta(seconds=ttl)
            return True
        return False
        
    async def touch(self, key: str) -> bool:
        """
        캐시 항목 접근 시간 갱신
        
        Args:
            key (str): 캐시 키
            
        Returns:
            bool: 성공 여부
        """
        entry = self.cache.get(key)
        if entry and not entry.is_expired():
            entry.access()
            self.cache.move_to_end(key)
            return True
        return False 