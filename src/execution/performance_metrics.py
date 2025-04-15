"""
실시간 성능 메트릭 수집기
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class PerformanceMetricsCollector:
    """실행 성능 메트릭을 수집하고 분석하는 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): 설정 정보
                - max_history_size: 최대 메트릭 이력 크기
                - metrics_weights: 메트릭 가중치
                - performance_thresholds: 성능 임계값
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 메트릭 저장소 초기화
        self.metrics_history = deque(maxlen=config.get('max_history_size', 1000))
        self.current_metrics = {
            'latency': 0.0,
            'fill_rate': 0.0,
            'slippage': 0.0,
            'execution_cost': 0.0,
            'success_rate': 0.0
        }
        
        # 실행 통계 초기화
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_latency': 0.0,
            'total_slippage': 0.0,
            'total_cost': 0.0
        }
        
        self.is_collecting = False
        self._collection_task = None
        
    async def initialize(self) -> None:
        """메트릭 수집기 초기화"""
        self.is_collecting = True
        self._collection_task = asyncio.create_task(self._collect_metrics())
        
    async def close(self) -> None:
        """메트릭 수집기 종료"""
        self.is_collecting = False
        if self._collection_task:
            await self._collection_task
            
    async def _collect_metrics(self) -> None:
        """주기적으로 메트릭을 수집하는 태스크"""
        while self.is_collecting:
            try:
                # 현재 메트릭 업데이트
                self._update_current_metrics()
                
                # 메트릭 이력에 추가
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': self.current_metrics.copy()
                })
                
                await asyncio.sleep(1)  # 1초마다 업데이트
                
            except Exception as e:
                self.logger.error(f"메트릭 수집 중 오류 발생: {str(e)}")
                await asyncio.sleep(1)
                
    def _update_current_metrics(self) -> None:
        """현재 메트릭 업데이트"""
        if self.execution_stats['total_executions'] > 0:
            self.current_metrics['latency'] = (
                self.execution_stats['total_latency'] / 
                self.execution_stats['total_executions']
            )
            self.current_metrics['fill_rate'] = (
                self.execution_stats['successful_executions'] / 
                self.execution_stats['total_executions']
            )
            self.current_metrics['slippage'] = (
                self.execution_stats['total_slippage'] / 
                self.execution_stats['total_executions']
            )
            self.current_metrics['execution_cost'] = (
                self.execution_stats['total_cost'] / 
                self.execution_stats['total_executions']
            )
            self.current_metrics['success_rate'] = (
                self.execution_stats['successful_executions'] / 
                self.execution_stats['total_executions']
            )
            
    def add_execution_metrics(self, metrics: Dict[str, float]) -> None:
        """
        실행 메트릭 추가
        
        Args:
            metrics (Dict[str, float]): 실행 메트릭
                - latency: 지연 시간
                - fill_rate: 체결률
                - slippage: 슬리피지
                - execution_cost: 실행 비용
                - success: 성공 여부
        """
        try:
            # 실행 통계 업데이트
            self.execution_stats['total_executions'] += 1
            if metrics.get('success', False):
                self.execution_stats['successful_executions'] += 1
            else:
                self.execution_stats['failed_executions'] += 1
                
            self.execution_stats['total_latency'] += metrics.get('latency', 0.0)
            self.execution_stats['total_slippage'] += metrics.get('slippage', 0.0)
            self.execution_stats['total_cost'] += metrics.get('execution_cost', 0.0)
            
            # 현재 메트릭 업데이트
            self._update_current_metrics()
            
            # 메트릭 이력에 추가
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': {
                    'latency': metrics.get('latency', 0.0),
                    'fill_rate': metrics.get('fill_rate', 0.0),
                    'slippage': metrics.get('slippage', 0.0),
                    'execution_cost': metrics.get('execution_cost', 0.0),
                    'success_rate': self.current_metrics['success_rate']
                }
            })
            
        except Exception as e:
            self.logger.error(f"실행 메트릭 추가 중 오류 발생: {str(e)}")
            
    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        메트릭 이력 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            List[Dict[str, Any]]: 메트릭 이력
        """
        history = list(self.metrics_history)
        
        if start_time:
            history = [h for h in history if h['timestamp'] >= start_time]
        if end_time:
            history = [h for h in history if h['timestamp'] <= end_time]
            
        return history
        
    def get_performance_score(self) -> float:
        """
        성능 점수 계산
        
        Returns:
            float: 성능 점수 (0.0 ~ 1.0)
        """
        weights = self.config.get('metrics_weights', {
            'latency': 0.2,
            'fill_rate': 0.3,
            'slippage': 0.2,
            'execution_cost': 0.2,
            'success_rate': 0.1
        })
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in self.current_metrics:
                # 메트릭 값을 0~1 범위로 정규화
                normalized_value = self._normalize_metric(metric, self.current_metrics[metric])
                score += normalized_value * weight
                
        return min(max(score, 0.0), 1.0)
        
    def _normalize_metric(self, metric: str, value: float) -> float:
        """
        메트릭 값을 0~1 범위로 정규화
        
        Args:
            metric (str): 메트릭 이름
            value (float): 메트릭 값
            
        Returns:
            float: 정규화된 값
        """
        thresholds = self.config.get('performance_thresholds', {
            'latency': {'min': 0.0, 'max': 1.0},
            'fill_rate': {'min': 0.0, 'max': 1.0},
            'slippage': {'min': 0.0, 'max': 0.1},
            'execution_cost': {'min': 0.0, 'max': 0.1},
            'success_rate': {'min': 0.0, 'max': 1.0}
        })
        
        if metric in thresholds:
            min_val = thresholds[metric]['min']
            max_val = thresholds[metric]['max']
            if max_val > min_val:
                return (value - min_val) / (max_val - min_val)
                
        return 0.0
