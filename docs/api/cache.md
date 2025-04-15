# 캐시 시스템 API 문서

## 1. 개요
캐시 시스템은 메모리, Redis, Memcache를 지원하는 캐시 관리 시스템입니다.

## 2. 초기화
```python
from src.cache.cache_manager import CacheManager

cache = CacheManager(
    config_dir="./config",
    data_dir="./data"
)
```

## 3. 메서드

### 3.1 시작/중지
```python
# 캐시 시작
cache.start()

# 캐시 중지
cache.stop()
```

### 3.2 캐시 조회
```python
# 캐시 조회
value = cache.get(
    key="my_key",
    cache_type="memory"  # "memory", "redis", "memcache"
)
```

### 3.3 캐시 저장
```python
# 캐시 저장
cache.set(
    key="my_key",
    value="my_value",
    ttl=3600,  # TTL (초)
    cache_type="memory"  # "memory", "redis", "memcache"
)
```

### 3.4 캐시 삭제
```python
# 캐시 삭제
cache.delete(
    key="my_key",
    cache_type="memory"  # "memory", "redis", "memcache"
)
```

### 3.5 캐시 초기화
```python
# 캐시 초기화
cache.clear(
    cache_type="memory"  # "memory", "redis", "memcache"
)
```

## 4. 통계

### 4.1 통계 조회
```python
# 캐시 히트 수
hits = cache.get_hit_count()

# 캐시 미스 수
misses = cache.get_miss_count()

# 캐시 제거 수
evictions = cache.get_eviction_count()

# 캐시 만료 수
expirations = cache.get_expiration_count()

# 전체 통계
stats = cache.get_stats()
```

### 4.2 통계 초기화
```python
# 통계 초기화
cache.reset_stats()
```

## 5. 설정

### 5.1 기본 설정
```json
{
    "default": {
        "type": "memory",
        "host": "localhost",
        "port": 6379,
        "password": "password",
        "db": 0,
        "ttl": 3600
    }
}
```

## 6. 에러 처리
- 모든 메서드는 예외를 발생시킬 수 있습니다.
- 에러는 로그에 기록됩니다.
- 통계에 에러 수가 기록됩니다.

## 7. 주의사항
1. 캐시 시작 전에 설정 파일이 있어야 합니다.
2. 메모리 캐시는 프로세스 내에서만 유효합니다.
3. Redis와 Memcache는 별도의 서버가 필요합니다. 