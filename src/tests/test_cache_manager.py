"""
캐시 관리자 테스트
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from src.execution.cache_manager import CacheManager, CacheEntry

@pytest.fixture
def cache_manager():
    """캐시 관리자 픽스처"""
    config = {
        'cache': {
            'max_entries': 100,
            'max_size': 1024 * 1024,  # 1MB
            'default_ttl': 60,  # 1분
            'cleanup_interval': 1  # 1초
        }
    }
    return CacheManager(config)

@pytest.mark.asyncio
async def test_cache_basic_operations(cache_manager):
    """기본 캐시 작업 테스트"""
    # 초기화
    await cache_manager.initialize()
    
    # 캐시 설정
    key = "test_key"
    value = {"data": "test_value"}
    
    success = await cache_manager.set(key, value)
    assert success is True
    
    # 캐시 조회
    cached_value = await cache_manager.get(key)
    assert cached_value == value
    
    # 캐시 삭제
    success = await cache_manager.delete(key)
    assert success is True
    
    # 삭제 후 조회
    cached_value = await cache_manager.get(key)
    assert cached_value is None
    
    # 정리
    await cache_manager.close()

@pytest.mark.asyncio
async def test_cache_expiration(cache_manager):
    """캐시 만료 테스트"""
    await cache_manager.initialize()
    
    # 짧은 TTL로 캐시 설정
    key = "expiring_key"
    value = {"data": "expiring_value"}
    ttl = 1  # 1초
    
    await cache_manager.set(key, value, ttl)
    
    # TTL 이전 조회
    cached_value = await cache_manager.get(key)
    assert cached_value == value
    
    # TTL 대기
    await asyncio.sleep(1.1)
    
    # TTL 이후 조회
    cached_value = await cache_manager.get(key)
    assert cached_value is None
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_cache_size_limit(cache_manager):
    """캐시 크기 제한 테스트"""
    await cache_manager.initialize()
    
    # 큰 데이터 생성
    large_value = "x" * (1024 * 1024)  # 1MB
    
    # 첫 번째 항목 설정
    success = await cache_manager.set("large_key1", large_value)
    assert success is True
    
    # 두 번째 항목 설정 시도 (실패 예상)
    success = await cache_manager.set("large_key2", large_value)
    assert success is False
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_cache_decorator(cache_manager):
    """캐시 데코레이터 테스트"""
    await cache_manager.initialize()
    
    call_count = 0
    
    @cache_manager.cache_execution(ttl=60)
    async def test_function(arg1, arg2):
        nonlocal call_count
        call_count += 1
        return arg1 + arg2
        
    # 첫 번째 호출
    result1 = await test_function(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    # 두 번째 호출 (캐시 사용)
    result2 = await test_function(1, 2)
    assert result2 == 3
    assert call_count == 1  # 함수가 다시 호출되지 않음
    
    # 다른 인자로 호출
    result3 = await test_function(2, 3)
    assert result3 == 5
    assert call_count == 2  # 새로운 인자로 인한 함수 호출
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_cache_cleanup(cache_manager):
    """캐시 청소 테스트"""
    await cache_manager.initialize()
    
    # 여러 항목 설정 (일부는 짧은 TTL)
    await cache_manager.set("key1", "value1", ttl=1)
    await cache_manager.set("key2", "value2", ttl=60)
    
    # 초기 상태 확인
    assert len(cache_manager.cache) == 2
    
    # TTL 대기
    await asyncio.sleep(1.1)
    
    # 청소 작업 실행
    await cache_manager._cleanup_expired()
    
    # 정리 후 상태 확인
    assert len(cache_manager.cache) == 1
    assert await cache_manager.get("key1") is None
    assert await cache_manager.get("key2") == "value2"
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_cache_statistics(cache_manager):
    """캐시 통계 테스트"""
    await cache_manager.initialize()
    
    # 캐시 작업 수행
    await cache_manager.set("key1", "value1")
    await cache_manager.get("key1")  # hit
    await cache_manager.get("key2")  # miss
    
    # 통계 확인
    stats = cache_manager.get_stats()
    assert stats['hits'] == 1
    assert stats['misses'] == 1
    assert stats['entries'] == 1
    assert stats['hit_ratio'] == 0.5
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_pattern_matching(cache_manager):
    """패턴 매칭 테스트"""
    await cache_manager.initialize()
    
    # 테스트 데이터 설정
    await cache_manager.set("test:1", "value1")
    await cache_manager.set("test:2", "value2")
    await cache_manager.set("other:1", "value3")
    
    # 패턴으로 항목 조회
    entries = await cache_manager.get_entries_by_pattern(r"test:\d+")
    assert len(entries) == 2
    assert ("test:1", "value1") in entries
    assert ("test:2", "value2") in entries
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_ttl_update(cache_manager):
    """TTL 업데이트 테스트"""
    await cache_manager.initialize()
    
    # 캐시 설정
    key = "update_ttl_key"
    await cache_manager.set(key, "value", ttl=1)
    
    # TTL 업데이트
    success = await cache_manager.update_ttl(key, 60)
    assert success is True
    
    # 원래 만료 시간 이후에도 접근 가능
    await asyncio.sleep(1.1)
    value = await cache_manager.get(key)
    assert value == "value"
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_touch_operation(cache_manager):
    """접근 시간 갱신 테스트"""
    await cache_manager.initialize()
    
    # 캐시 설정
    key = "touch_key"
    await cache_manager.set(key, "value")
    
    # 초기 접근 시간 기록
    entry = cache_manager.cache[key]
    initial_access_time = entry.last_accessed
    
    # 잠시 대기
    await asyncio.sleep(0.1)
    
    # 접근 시간 갱신
    success = await cache_manager.touch(key)
    assert success is True
    
    # 갱신된 접근 시간 확인
    assert entry.last_accessed > initial_access_time
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_concurrent_access(cache_manager):
    """동시 접근 테스트"""
    await cache_manager.initialize()
    
    async def concurrent_operation(key: str, value: Any):
        await cache_manager.set(key, value)
        await asyncio.sleep(0.1)
        return await cache_manager.get(key)
        
    # 여러 작업 동시 실행
    tasks = [
        concurrent_operation(f"key{i}", f"value{i}")
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 결과 검증
    assert len(results) == 10
    assert all(
        results[i] == f"value{i}"
        for i in range(10)
    )
    
    await cache_manager.close()

@pytest.mark.asyncio
async def test_error_handling(cache_manager):
    """오류 처리 테스트"""
    await cache_manager.initialize()
    
    # 잘못된 TTL 값으로 설정 시도
    with pytest.raises(ValueError):
        await cache_manager.set("key", "value", ttl=-1)
        
    # 존재하지 않는 키 삭제
    success = await cache_manager.delete("nonexistent_key")
    assert success is False
    
    # 존재하지 않는 키의 TTL 업데이트
    success = await cache_manager.update_ttl("nonexistent_key", 60)
    assert success is False
    
    await cache_manager.close() 