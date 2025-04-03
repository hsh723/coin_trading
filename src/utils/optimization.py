"""
메모리 및 성능 최적화 유틸리티 모듈
"""

import gc
import psutil
import logging
from typing import Optional, List, Dict, Any
import time
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, memory_threshold: float = 0.8):
        """
        메모리 관리자 초기화
        
        Args:
            memory_threshold (float): 메모리 사용량 임계값 (0.0 ~ 1.0)
        """
        self.memory_threshold = memory_threshold
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """
        현재 메모리 사용량 조회
        
        Returns:
            float: 메모리 사용량 (0.0 ~ 1.0)
        """
        return self.process.memory_percent() / 100.0
    
    def is_memory_critical(self) -> bool:
        """
        메모리 사용량이 임계값을 초과하는지 확인
        
        Returns:
            bool: 메모리 사용량이 임계값을 초과하는지 여부
        """
        return self.get_memory_usage() > self.memory_threshold
    
    def optimize_memory(self) -> None:
        """메모리 최적화"""
        if self.is_memory_critical():
            logger.warning("메모리 사용량이 임계값을 초과하여 최적화를 시작합니다.")
            
            # 가비지 컬렉션 실행
            gc.collect()
            
            # 메모리 사용량 재확인
            if self.is_memory_critical():
                logger.error("메모리 최적화 후에도 사용량이 높습니다.")
                raise MemoryError("메모리 사용량이 임계값을 초과합니다.")

class PerformanceMonitor:
    def __init__(self):
        """성능 모니터 초기화"""
        self.start_time = time.time()
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
    
    def record_operation(self, operation: str, duration: float) -> None:
        """
        작업 실행 시간 기록
        
        Args:
            operation (str): 작업 이름
            duration (float): 실행 시간 (초)
        """
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
    
    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """
        작업 통계 조회
        
        Args:
            operation (str): 작업 이름
            
        Returns:
            Dict[str, float]: 작업 통계
        """
        if operation not in self.operation_times:
            return {}
        
        times = self.operation_times[operation]
        return {
            'count': self.operation_counts[operation],
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times)
        }

def measure_performance(operation: str):
    """
    성능 측정 데코레이터
    
    Args:
        operation (str): 작업 이름
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_operation(operation, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_operation(operation, duration)
                raise
        return wrapper
    return decorator

class DataOptimizer:
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임 최적화
        
        Args:
            df (pd.DataFrame): 최적화할 데이터프레임
            
        Returns:
            pd.DataFrame: 최적화된 데이터프레임
        """
        # 메모리 사용량 최적화
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        
        return df
    
    @staticmethod
    def optimize_list(data: List[Any]) -> List[Any]:
        """
        리스트 최적화
        
        Args:
            data (List[Any]): 최적화할 리스트
            
        Returns:
            List[Any]: 최적화된 리스트
        """
        # 중복 제거 및 정렬
        return sorted(set(data))

# 전역 인스턴스
memory_manager = MemoryManager()
performance_monitor = PerformanceMonitor()
data_optimizer = DataOptimizer() 