"""
오류 복구 메커니즘 모듈
"""

import asyncio
import logging
from typing import Callable, Any, Optional
from functools import wraps
import time
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class RecoveryManager:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        복구 관리자 초기화
        
        Args:
            max_retries (int): 최대 재시도 횟수
            retry_delay (float): 재시도 간격 (초)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts = {}
        self.last_error_times = {}
    
    def should_retry(self, operation: str) -> bool:
        """
        재시도 여부 확인
        
        Args:
            operation (str): 작업 이름
            
        Returns:
            bool: 재시도 여부
        """
        current_time = time.time()
        
        # 에러 카운트 초기화 (1시간 경과)
        if operation in self.last_error_times:
            if current_time - self.last_error_times[operation] > 3600:
                self.error_counts[operation] = 0
        
        # 에러 카운트 증가
        self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
        self.last_error_times[operation] = current_time
        
        return self.error_counts[operation] <= self.max_retries
    
    def reset_error_count(self, operation: str) -> None:
        """
        에러 카운트 초기화
        
        Args:
            operation (str): 작업 이름
        """
        if operation in self.error_counts:
            del self.error_counts[operation]
        if operation in self.last_error_times:
            del self.last_error_times[operation]

def with_recovery(operation: str, recovery_manager: Optional[RecoveryManager] = None):
    """
    복구 메커니즘 데코레이터
    
    Args:
        operation (str): 작업 이름
        recovery_manager (Optional[RecoveryManager]): 복구 관리자
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            manager = recovery_manager or RecoveryManager()
            last_error = None
            
            while manager.should_retry(operation):
                try:
                    result = await func(*args, **kwargs)
                    manager.reset_error_count(operation)
                    return result
                except Exception as e:
                    last_error = e
                    error_msg = f"{operation} 실행 중 오류 발생: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    
                    # 재시도 대기
                    await asyncio.sleep(manager.retry_delay)
            
            # 최대 재시도 횟수 초과
            error_msg = f"{operation} 최대 재시도 횟수 초과. 마지막 오류: {str(last_error)}"
            logger.error(error_msg)
            raise last_error
        
        return wrapper
    return decorator

class SystemMonitor:
    def __init__(self):
        """시스템 모니터 초기화"""
        self.start_time = datetime.now()
        self.error_count = 0
        self.last_error_time = None
        self.uptime = 0
    
    def record_error(self) -> None:
        """에러 기록"""
        self.error_count += 1
        self.last_error_time = datetime.now()
    
    def get_system_status(self) -> dict:
        """
        시스템 상태 조회
        
        Returns:
            dict: 시스템 상태 정보
        """
        current_time = datetime.now()
        self.uptime = (current_time - self.start_time).total_seconds()
        
        return {
            'uptime': self.uptime,
            'error_count': self.error_count,
            'last_error_time': self.last_error_time,
            'is_healthy': self.error_count < 10  # 에러가 10회 미만이면 정상으로 간주
        }

# 전역 복구 관리자 및 시스템 모니터 인스턴스
recovery_manager = RecoveryManager()
system_monitor = SystemMonitor() 