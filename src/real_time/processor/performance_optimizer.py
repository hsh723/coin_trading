"""
실시간 처리 시스템 성능 최적화 모듈
"""

import asyncio
import psutil
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    io_wait: float
    network_latency: float
    event_queue_size: int
    processing_time: float

class PerformanceOptimizer:
    def __init__(self, config: Dict = None):
        """
        성능 최적화 관리자 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config or {
            'cpu_threshold': 80.0,  # CPU 사용률 임계값
            'memory_threshold': 85.0,  # 메모리 사용률 임계값
            'queue_size_threshold': 1000,  # 큐 크기 임계값
            'latency_threshold': 0.1,  # 지연 시간 임계값 (초)
            'optimization_interval': 1.0  # 최적화 주기 (초)
        }
        
        self.metrics_history: List[SystemMetrics] = []
        self.optimization_tasks = []
        
    async def start_monitoring(self):
        """성능 모니터링 시작"""
        while True:
            try:
                metrics = await self._collect_metrics()
                await self._analyze_performance(metrics)
                await self._optimize_if_needed(metrics)
                
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                    
            except Exception as e:
                logger.error(f"성능 모니터링 중 오류 발생: {str(e)}")
                
            await asyncio.sleep(self.config['optimization_interval'])
            
    async def _collect_metrics(self) -> SystemMetrics:
        """시스템 메트릭스 수집"""
        return SystemMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            io_wait=psutil.cpu_times_percent().iowait,
            network_latency=await self._measure_network_latency(),
            event_queue_size=len(asyncio.all_tasks()),
            processing_time=await self._measure_processing_time()
        )
        
    async def _analyze_performance(self, metrics: SystemMetrics):
        """성능 분석"""
        if metrics.cpu_usage > self.config['cpu_threshold']:
            await self._optimize_cpu_usage()
            
        if metrics.memory_usage > self.config['memory_threshold']:
            await self._optimize_memory_usage()
            
        if metrics.event_queue_size > self.config['queue_size_threshold']:
            await self._optimize_queue_size()
            
        if metrics.network_latency > self.config['latency_threshold']:
            await self._optimize_network_latency()
            
    async def _optimize_if_needed(self, metrics: SystemMetrics):
        """필요한 경우 최적화 수행"""
        optimizations = []
        
        if metrics.cpu_usage > self.config['cpu_threshold']:
            optimizations.append(self._optimize_cpu_usage())
            
        if metrics.memory_usage > self.config['memory_threshold']:
            optimizations.append(self._optimize_memory_usage())
            
        if optimizations:
            await asyncio.gather(*optimizations)
            
    async def _optimize_cpu_usage(self):
        """CPU 사용률 최적화"""
        try:
            # 우선순위가 낮은 작업 일시 중단
            for task in self.optimization_tasks:
                if not task.done():
                    task.cancel()
                    
            # 작업 스케줄링 최적화
            current_process = psutil.Process()
            current_process.nice(10)  # 우선순위 낮춤
            
            logger.info("CPU 사용률 최적화 완료")
            
        except Exception as e:
            logger.error(f"CPU 최적화 중 오류 발생: {str(e)}")
            
    async def _optimize_memory_usage(self):
        """메모리 사용률 최적화"""
        try:
            # 가비지 컬렉션 강제 실행
            import gc
            gc.collect()
            
            # 메모리 캐시 정리
            if hasattr(self, 'cache'):
                self.cache.clear()
                
            logger.info("메모리 사용률 최적화 완료")
            
        except Exception as e:
            logger.error(f"메모리 최적화 중 오류 발생: {str(e)}")
            
    async def _optimize_queue_size(self):
        """이벤트 큐 크기 최적화"""
        try:
            # 오래된 이벤트 제거
            current_time = datetime.now()
            for task in asyncio.all_tasks():
                if (current_time - task.get_coro().cr_frame.f_locals.get('start_time', current_time)).total_seconds() > 60:
                    task.cancel()
                    
            logger.info("이벤트 큐 크기 최적화 완료")
            
        except Exception as e:
            logger.error(f"큐 최적화 중 오류 발생: {str(e)}")
            
    async def _optimize_network_latency(self):
        """네트워크 지연 시간 최적화"""
        try:
            # 네트워크 버퍼 크기 조정
            import socket
            socket.setdefaulttimeout(5.0)
            
            logger.info("네트워크 지연 시간 최적화 완료")
            
        except Exception as e:
            logger.error(f"네트워크 최적화 중 오류 발생: {str(e)}")
            
    async def _measure_network_latency(self) -> float:
        """네트워크 지연 시간 측정"""
        try:
            start_time = asyncio.get_event_loop().time()
            # 여기에 실제 네트워크 요청 테스트 추가
            await asyncio.sleep(0.001)  # 더미 지연
            return asyncio.get_event_loop().time() - start_time
        except Exception:
            return 0.0
            
    async def _measure_processing_time(self) -> float:
        """처리 시간 측정"""
        try:
            start_time = asyncio.get_event_loop().time()
            # 여기에 실제 처리 시간 측정 로직 추가
            await asyncio.sleep(0.001)  # 더미 처리
            return asyncio.get_event_loop().time() - start_time
        except Exception:
            return 0.0 