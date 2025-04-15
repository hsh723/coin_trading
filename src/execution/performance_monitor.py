"""
실행 시스템 성능 모니터링 모듈
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np

from src.execution.logger import ExecutionLogger

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """실행 시스템 성능 모니터"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        성능 모니터 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        
        # 성능 지표 설정
        self.latency_threshold = config['performance']['latency_threshold']
        self.success_rate_threshold = config['performance']['success_rate_threshold']
        self.fill_rate_threshold = config['performance']['fill_rate_threshold']
        
        # 성능 데이터 저장소
        self.latency_data = deque(maxlen=1000)  # 지연시간 데이터
        self.execution_data = deque(maxlen=1000)  # 실행 데이터
        self.fill_data = deque(maxlen=1000)  # 체결률 데이터
        
        # 성능 통계
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_orders': 0,
            'filled_orders': 0,
            'total_volume': 0.0,
            'filled_volume': 0.0
        }
        
        # 로거 초기화
        self.logger = ExecutionLogger(config)
        
    async def initialize(self):
        """성능 모니터 초기화"""
        try:
            # 로거 초기화
            await self.logger.initialize()
            
            logger.info("성능 모니터 초기화 완료")
            
        except Exception as e:
            logger.error(f"성능 모니터 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            # 로거 정리
            await self.logger.close()
            
            logger.info("성능 모니터 종료")
            
        except Exception as e:
            logger.error(f"성능 모니터 종료 실패: {str(e)}")
            
    def record_latency(self, latency: float) -> None:
        """
        지연시간 기록
        
        Args:
            latency (float): 지연시간 (밀리초)
        """
        self.latency_data.append({
            'timestamp': datetime.now(),
            'latency': latency
        })
        
        # 로그 기록
        self.logger.log_performance({
            'type': 'latency',
            'value': latency
        })
        
        # 임계값 초과 확인
        if latency > self.latency_threshold:
            logger.warning(f"지연시간 임계값 초과: {latency}ms")
            
    def record_execution(
        self,
        execution_data: Dict[str, Any]
    ) -> None:
        """
        실행 결과 기록
        
        Args:
            execution_data (Dict[str, Any]): 실행 데이터
        """
        self.execution_data.append({
            'timestamp': datetime.now(),
            'data': execution_data
        })
        
        # 통계 업데이트
        self.stats['total_executions'] += 1
        if execution_data.get('success', False):
            self.stats['successful_executions'] += 1
        else:
            self.stats['failed_executions'] += 1
            
        # 로그 기록
        self.logger.log_performance({
            'type': 'execution',
            'data': execution_data
        })
        
        # 성공률 확인
        success_rate = self.get_success_rate()
        if success_rate < self.success_rate_threshold:
            logger.warning(f"실행 성공률 임계값 미달: {success_rate:.2%}")
            
    def record_fill(
        self,
        order_data: Dict[str, Any]
    ) -> None:
        """
        주문 체결 기록
        
        Args:
            order_data (Dict[str, Any]): 주문 데이터
        """
        self.fill_data.append({
            'timestamp': datetime.now(),
            'data': order_data
        })
        
        # 통계 업데이트
        self.stats['total_orders'] += 1
        self.stats['total_volume'] += float(order_data.get('quantity', 0))
        
        if order_data.get('filled', False):
            self.stats['filled_orders'] += 1
            self.stats['filled_volume'] += float(order_data.get('filled_quantity', 0))
            
        # 로그 기록
        self.logger.log_performance({
            'type': 'fill',
            'data': order_data
        })
        
        # 체결률 확인
        fill_rate = self.get_fill_rate()
        if fill_rate < self.fill_rate_threshold:
            logger.warning(f"주문 체결률 임계값 미달: {fill_rate:.2%}")
            
    def get_latency_stats(
        self,
        window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """
        지연시간 통계 조회
        
        Args:
            window (Optional[timedelta]): 시간 범위
            
        Returns:
            Dict[str, float]: 지연시간 통계
        """
        try:
            # 데이터 필터링
            if window:
                cutoff = datetime.now() - window
                data = [
                    d['latency']
                    for d in self.latency_data
                    if d['timestamp'] >= cutoff
                ]
            else:
                data = [d['latency'] for d in self.latency_data]
                
            if not data:
                return {
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                    'median': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                }
                
            return {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'p95': float(np.percentile(data, 95)),
                'p99': float(np.percentile(data, 99))
            }
            
        except Exception as e:
            logger.error(f"지연시간 통계 조회 실패: {str(e)}")
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
            
    def get_success_rate(
        self,
        window: Optional[timedelta] = None
    ) -> float:
        """
        실행 성공률 조회
        
        Args:
            window (Optional[timedelta]): 시간 범위
            
        Returns:
            float: 성공률
        """
        try:
            # 데이터 필터링
            if window:
                cutoff = datetime.now() - window
                data = [
                    d['data']
                    for d in self.execution_data
                    if d['timestamp'] >= cutoff
                ]
            else:
                data = [d['data'] for d in self.execution_data]
                
            if not data:
                return 1.0
                
            success_count = sum(1 for d in data if d.get('success', False))
            return success_count / len(data)
            
        except Exception as e:
            logger.error(f"실행 성공률 조회 실패: {str(e)}")
            return 1.0
            
    def get_fill_rate(
        self,
        window: Optional[timedelta] = None
    ) -> float:
        """
        주문 체결률 조회
        
        Args:
            window (Optional[timedelta]): 시간 범위
            
        Returns:
            float: 체결률
        """
        try:
            # 데이터 필터링
            if window:
                cutoff = datetime.now() - window
                data = [
                    d['data']
                    for d in self.fill_data
                    if d['timestamp'] >= cutoff
                ]
            else:
                data = [d['data'] for d in self.fill_data]
                
            if not data:
                return 1.0
                
            total_volume = sum(float(d.get('quantity', 0)) for d in data)
            filled_volume = sum(float(d.get('filled_quantity', 0)) for d in data)
            
            return filled_volume / total_volume if total_volume > 0 else 1.0
            
        except Exception as e:
            logger.error(f"주문 체결률 조회 실패: {str(e)}")
            return 1.0
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        성능 요약 조회
        
        Returns:
            Dict[str, Any]: 성능 요약
        """
        return {
            'latency': self.get_latency_stats(),
            'success_rate': self.get_success_rate(),
            'fill_rate': self.get_fill_rate(),
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
    def reset_stats(self) -> None:
        """통계 초기화"""
        self.latency_data.clear()
        self.execution_data.clear()
        self.fill_data.clear()
        
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_orders': 0,
            'filled_orders': 0,
            'total_volume': 0.0,
            'filled_volume': 0.0
        } 