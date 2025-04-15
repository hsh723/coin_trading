import os
import json
import logging
import threading
import queue
import time
import pickle
import redis
import memcache
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

class CacheManager:
    """캐시 클래스"""
    
    def __init__(self,
                 config_dir: str = "./config",
                 data_dir: str = "./data"):
        """
        캐시 초기화
        
        Args:
            config_dir: 설정 디렉토리
            data_dir: 데이터 디렉토리
        """
        self.config_dir = config_dir
        self.data_dir = data_dir
        
        # 로거 설정
        self.logger = logging.getLogger("cache")
        
        # 캐시 큐
        self.cache_queue = queue.Queue()
        
        # 디렉토리 생성
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 설정 로드
        self.config = self._load_config()
        
        # 메모리 캐시
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # 캐시 클라이언트
        self.clients: Dict[str, Any] = {}
        
        # 캐시 관리자
        self.is_running = False
        
        # 통계
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "errors": 0
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            config_path = os.path.join(self.config_dir, "cache_config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {
                "default": {
                    "type": "memory",
                    "ttl": 3600
                }
            }
            
    def start(self) -> None:
        """캐시 시작"""
        try:
            self.is_running = True
            
            # 캐시 처리 시작
            threading.Thread(target=self._process_cache, daemon=True).start()
            
            # 만료된 캐시 정리 시작
            threading.Thread(target=self._cleanup_expired, daemon=True).start()
            
            self.logger.info("캐시가 시작되었습니다")
            
        except Exception as e:
            self.logger.error(f"캐시 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """캐시 중지"""
        try:
            self.is_running = False
            
            # 클라이언트 종료
            for client in self.clients.values():
                self._close_client(client)
                
            self.logger.info("캐시가 중지되었습니다")
            
        except Exception as e:
            self.logger.error(f"캐시 중지 중 오류 발생: {e}")
            raise
            
    def _create_client(self,
                      cache_type: str,
                      config: Dict[str, Any]) -> Any:
        """
        캐시 클라이언트 생성
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            
        Returns:
            캐시 클라이언트
        """
        try:
            if cache_type == "redis":
                # Redis 클라이언트
                return redis.Redis(
                    host=config["host"],
                    port=config["port"],
                    password=config["password"],
                    decode_responses=False
                )
                
            elif cache_type == "memcache":
                # Memcache 클라이언트
                return memcache.Client(
                    [f"{config['host']}:{config['port']}"]
                )
                
            else:
                raise ValueError(f"지원하지 않는 캐시 타입: {cache_type}")
                
        except Exception as e:
            self.logger.error(f"캐시 클라이언트 생성 중 오류 발생: {e}")
            raise
            
    def _close_client(self, client: Any) -> None:
        """
        캐시 클라이언트 종료
        
        Args:
            client: 캐시 클라이언트
        """
        try:
            if isinstance(client, redis.Redis):
                client.close()
            elif isinstance(client, memcache.Client):
                client.disconnect_all()
                
        except Exception as e:
            self.logger.error(f"캐시 클라이언트 종료 중 오류 발생: {e}")
            
    def get_client(self,
                   cache_type: str,
                   config: Dict[str, Any]) -> Any:
        """
        캐시 클라이언트 가져오기
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            
        Returns:
            캐시 클라이언트
        """
        try:
            # 클라이언트 키 생성
            client_key = f"{cache_type}_{json.dumps(config, sort_keys=True)}"
            
            # 클라이언트 확인
            if client_key not in self.clients:
                self.clients[client_key] = self._create_client(cache_type, config)
                
            return self.clients[client_key]
            
        except Exception as e:
            self.logger.error(f"캐시 클라이언트 가져오기 중 오류 발생: {e}")
            raise
            
    def _process_cache(self) -> None:
        """캐시 처리 루프"""
        try:
            while self.is_running:
                if not self.cache_queue.empty():
                    cache = self.cache_queue.get()
                    self._handle_cache(
                        cache["cache_type"],
                        cache["config"],
                        cache["operation"],
                        cache["key"],
                        cache.get("value"),
                        cache.get("ttl")
                    )
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"캐시 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def _handle_cache(self,
                     cache_type: str,
                     config: Dict[str, Any],
                     operation: str,
                     key: str,
                     value: Optional[Any] = None,
                     ttl: Optional[int] = None) -> None:
        """
        캐시 작업 처리
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            operation: 작업 타입
            key: 캐시 키
            value: 캐시 값
            ttl: 만료 시간
        """
        try:
            if operation == "get":
                # 캐시 조회
                return self._get_cache(cache_type, config, key)
                
            elif operation == "set":
                # 캐시 저장
                self._set_cache(cache_type, config, key, value, ttl)
                
            elif operation == "delete":
                # 캐시 삭제
                self._delete_cache(cache_type, config, key)
                
            elif operation == "clear":
                # 캐시 초기화
                self._clear_cache(cache_type, config)
                
        except Exception as e:
            self.logger.error(f"캐시 작업 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _get_cache(self,
                  cache_type: str,
                  config: Dict[str, Any],
                  key: str) -> Optional[Any]:
        """
        캐시 조회
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            key: 캐시 키
            
        Returns:
            캐시 값
        """
        try:
            if cache_type == "memory":
                # 메모리 캐시 조회
                if key in self.memory_cache:
                    cache = self.memory_cache[key]
                    if cache["expires"] > time.time():
                        self.stats["hits"] += 1
                        return pickle.loads(cache["value"])
                    else:
                        del self.memory_cache[key]
                        self.stats["expirations"] += 1
                        
                self.stats["misses"] += 1
                return None
                
            elif cache_type == "redis":
                # Redis 캐시 조회
                client = self.get_client(cache_type, config)
                value = client.get(key)
                
                if value:
                    self.stats["hits"] += 1
                    return pickle.loads(value)
                    
                self.stats["misses"] += 1
                return None
                
            elif cache_type == "memcache":
                # Memcache 캐시 조회
                client = self.get_client(cache_type, config)
                value = client.get(key)
                
                if value:
                    self.stats["hits"] += 1
                    return pickle.loads(value)
                    
                self.stats["misses"] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"캐시 조회 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _set_cache(self,
                  cache_type: str,
                  config: Dict[str, Any],
                  key: str,
                  value: Any,
                  ttl: Optional[int] = None) -> None:
        """
        캐시 저장
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            key: 캐시 키
            value: 캐시 값
            ttl: 만료 시간
        """
        try:
            if ttl is None:
                ttl = config.get("ttl", 3600)
                
            if cache_type == "memory":
                # 메모리 캐시 저장
                self.memory_cache[key] = {
                    "value": pickle.dumps(value),
                    "expires": time.time() + ttl
                }
                
            elif cache_type == "redis":
                # Redis 캐시 저장
                client = self.get_client(cache_type, config)
                client.setex(key, ttl, pickle.dumps(value))
                
            elif cache_type == "memcache":
                # Memcache 캐시 저장
                client = self.get_client(cache_type, config)
                client.set(key, pickle.dumps(value), ttl)
                
        except Exception as e:
            self.logger.error(f"캐시 저장 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _delete_cache(self,
                     cache_type: str,
                     config: Dict[str, Any],
                     key: str) -> None:
        """
        캐시 삭제
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            key: 캐시 키
        """
        try:
            if cache_type == "memory":
                # 메모리 캐시 삭제
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    
            elif cache_type == "redis":
                # Redis 캐시 삭제
                client = self.get_client(cache_type, config)
                client.delete(key)
                
            elif cache_type == "memcache":
                # Memcache 캐시 삭제
                client = self.get_client(cache_type, config)
                client.delete(key)
                
        except Exception as e:
            self.logger.error(f"캐시 삭제 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _clear_cache(self,
                    cache_type: str,
                    config: Dict[str, Any]) -> None:
        """
        캐시 초기화
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
        """
        try:
            if cache_type == "memory":
                # 메모리 캐시 초기화
                self.memory_cache.clear()
                
            elif cache_type == "redis":
                # Redis 캐시 초기화
                client = self.get_client(cache_type, config)
                client.flushdb()
                
            elif cache_type == "memcache":
                # Memcache 캐시 초기화
                client = self.get_client(cache_type, config)
                client.flush_all()
                
        except Exception as e:
            self.logger.error(f"캐시 초기화 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _cleanup_expired(self) -> None:
        """만료된 캐시 정리"""
        try:
            while self.is_running:
                current_time = time.time()
                
                # 메모리 캐시 정리
                expired_keys = [
                    key for key, cache in self.memory_cache.items()
                    if cache["expires"] <= current_time
                ]
                
                for key in expired_keys:
                    del self.memory_cache[key]
                    self.stats["expirations"] += 1
                    
                time.sleep(60)
                
        except Exception as e:
            self.logger.error(f"만료된 캐시 정리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def get(self,
            cache_type: str,
            config: Dict[str, Any],
            key: str) -> Optional[Any]:
        """
        캐시 조회 요청
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            key: 캐시 키
            
        Returns:
            캐시 값
        """
        try:
            result_queue = queue.Queue()
            
            self.cache_queue.put({
                "cache_type": cache_type,
                "config": config,
                "operation": "get",
                "key": key,
                "result_queue": result_queue
            })
            
            return result_queue.get()
            
        except Exception as e:
            self.logger.error(f"캐시 조회 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def set(self,
            cache_type: str,
            config: Dict[str, Any],
            key: str,
            value: Any,
            ttl: Optional[int] = None) -> None:
        """
        캐시 저장 요청
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            key: 캐시 키
            value: 캐시 값
            ttl: 만료 시간
        """
        try:
            self.cache_queue.put({
                "cache_type": cache_type,
                "config": config,
                "operation": "set",
                "key": key,
                "value": value,
                "ttl": ttl
            })
            
        except Exception as e:
            self.logger.error(f"캐시 저장 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def delete(self,
              cache_type: str,
              config: Dict[str, Any],
              key: str) -> None:
        """
        캐시 삭제 요청
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
            key: 캐시 키
        """
        try:
            self.cache_queue.put({
                "cache_type": cache_type,
                "config": config,
                "operation": "delete",
                "key": key
            })
            
        except Exception as e:
            self.logger.error(f"캐시 삭제 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def clear(self,
             cache_type: str,
             config: Dict[str, Any]) -> None:
        """
        캐시 초기화 요청
        
        Args:
            cache_type: 캐시 타입
            config: 클라이언트 설정
        """
        try:
            self.cache_queue.put({
                "cache_type": cache_type,
                "config": config,
                "operation": "clear"
            })
            
        except Exception as e:
            self.logger.error(f"캐시 초기화 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def get_hits_count(self) -> int:
        """
        캐시 히트 수 조회
        
        Returns:
            캐시 히트 수
        """
        return self.stats["hits"]
        
    def get_misses_count(self) -> int:
        """
        캐시 미스 수 조회
        
        Returns:
            캐시 미스 수
        """
        return self.stats["misses"]
        
    def get_evictions_count(self) -> int:
        """
        캐시 제거 수 조회
        
        Returns:
            캐시 제거 수
        """
        return self.stats["evictions"]
        
    def get_expirations_count(self) -> int:
        """
        캐시 만료 수 조회
        
        Returns:
            캐시 만료 수
        """
        return self.stats["expirations"]
        
    def get_stats(self) -> Dict[str, int]:
        """
        캐시 통계 조회
        
        Returns:
            캐시 통계
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """캐시 통계 초기화"""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "errors": 0
        } 