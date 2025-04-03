"""
성능 최적화 유틸리티 모듈
"""

import pandas as pd
import numpy as np
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import cProfile
import pstats
import io
import logging
from typing import List, Dict, Any, Callable, Union
import gc

logger = logging.getLogger(__name__)

class DataFrameOptimizer:
    """데이터프레임 처리 최적화 클래스"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임의 데이터 타입을 최적화"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'object':
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        return df
    
    @staticmethod
    def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임의 메모리 사용량 감소"""
        start_mem = df.memory_usage().sum() / 1024**2
        df = DataFrameOptimizer.optimize_dtypes(df)
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f'메모리 사용량 감소: {start_mem:.2f}MB -> {end_mem:.2f}MB')
        return df
    
    @staticmethod
    def chunk_processing(df: pd.DataFrame, chunk_size: int = 10000) -> pd.DataFrame:
        """대용량 데이터프레임을 청크 단위로 처리"""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunk = DataFrameOptimizer.optimize_dtypes(chunk)
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

class MemoryMonitor:
    """메모리 사용량 모니터링 및 최적화 클래스"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """현재 프로세스의 메모리 사용량 조회"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024**2,  # MB
            'vms': memory_info.vms / 1024**2,  # MB
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def optimize_memory():
        """메모리 최적화 수행"""
        gc.collect()  # 가비지 컬렉션 실행
        
        # 메모리 사용량 로깅
        memory_usage = MemoryMonitor.get_memory_usage()
        logger.info(f'메모리 사용량: {memory_usage["rss"]:.2f}MB')
        
        if memory_usage['percent'] > 80:
            logger.warning('메모리 사용량이 높습니다. 최적화가 필요합니다.')
    
    @staticmethod
    def clear_cache():
        """캐시 메모리 정리"""
        gc.collect()
        pd.DataFrame._clear_item_cache()
        pd.DataFrame._clear_item_cache()

class ParallelProcessor:
    """병렬 처리 헬퍼 클래스"""
    
    @staticmethod
    def parallel_apply(
        df: pd.DataFrame,
        func: Callable,
        num_processes: int = None,
        chunk_size: int = 1000
    ) -> pd.DataFrame:
        """데이터프레임에 함수를 병렬로 적용"""
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        
        # 데이터프레임을 청크로 분할
        chunks = np.array_split(df, num_processes)
        
        # 멀티프로세싱으로 처리
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(func, chunks))
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def parallel_map(
        items: List[Any],
        func: Callable,
        num_threads: int = None
    ) -> List[Any]:
        """리스트의 각 항목에 함수를 병렬로 적용"""
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(func, items))
        
        return results

class CacheManager:
    """계산 결과 캐싱 관리 클래스"""
    
    @staticmethod
    def cached_function(
        maxsize: int = 128,
        typed: bool = False
    ) -> Callable:
        """함수 결과를 캐싱하는 데코레이터"""
        return lru_cache(maxsize=maxsize, typed=typed)
    
    @staticmethod
    def clear_all_caches():
        """모든 캐시 초기화"""
        CacheManager.cached_function.cache_clear()
        gc.collect()

class Profiler:
    """프로파일링 도구 클래스"""
    
    @staticmethod
    def profile_function(func: Callable) -> Callable:
        """함수 프로파일링 데코레이터"""
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            try:
                return pr.runcall(func, *args, **kwargs)
            finally:
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats()
                logger.info(f'프로파일링 결과:\n{s.getvalue()}')
        return wrapper
    
    @staticmethod
    def profile_code(code: str):
        """코드 블록 프로파일링"""
        pr = cProfile.Profile()
        pr.enable()
        exec(code)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        logger.info(f'프로파일링 결과:\n{s.getvalue()}')

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임 최적화 헬퍼 함수"""
    return DataFrameOptimizer.reduce_memory(df)

def monitor_memory():
    """메모리 모니터링 헬퍼 함수"""
    return MemoryMonitor.get_memory_usage()

def parallel_process(
    items: List[Any],
    func: Callable,
    num_workers: int = None
) -> List[Any]:
    """병렬 처리 헬퍼 함수"""
    return ParallelProcessor.parallel_map(items, func, num_workers)

@CacheManager.cached_function(maxsize=128)
def cached_calculation(func: Callable, *args, **kwargs) -> Any:
    """계산 결과 캐싱 헬퍼 함수"""
    return func(*args, **kwargs)

def profile_code_block(code: str):
    """코드 블록 프로파일링 헬퍼 함수"""
    Profiler.profile_code(code) 