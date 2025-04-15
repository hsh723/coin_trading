"""
실행 모니터링 시스템

이 모듈은 주문 실행 과정을 모니터링하고 성능 메트릭을 수집합니다.
주요 기능:
- 실행 지연시간 모니터링
- 체결률 모니터링
- 슬리피지 모니터링
- 실행 비용 모니터링
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class ExecutionMonitor:
    """실행 모니터"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        self.config = config
        self.metrics = {
            'latency': {
                'current': 0.0,
                'average': 0.0,
                'max': 0.0,
                'threshold': config.get('latency_threshold', 1.0),
                'is_healthy': True
            },
            'fill_rate': {
                'current': 0.0,
                'average': 0.0,
                'min': 1.0,
                'threshold': config.get('fill_rate_threshold', 0.95),
                'is_healthy': True
            },
            'slippage': {
                'current': 0.0,
                'average': 0.0,
                'max': 0.0,
                'threshold': config.get('slippage_threshold', 0.001),
                'is_healthy': True
            },
            'cost': {
                'current': 0.0,
                'average': 0.0,
                'max': 0.0,
                'threshold': config.get('cost_threshold', 0.002),
                'is_healthy': True
            }
        }
        self.logger = logging.getLogger(__name__)
        
        # 모니터링 설정
        self.update_interval = config.get('update_interval', 1.0)  # 기본 1초
        self.window_size = config.get('window_size', 100)  # 기본 100개 샘플
        self.latency_threshold = config.get('latency_threshold', 1.0)  # 기본 1초
        self.fill_rate_threshold = config.get('fill_rate_threshold', 0.95)  # 기본 95%
        self.slippage_threshold = config.get('slippage_threshold', 0.001)  # 기본 0.1%
        self.cost_threshold = config.get('cost_threshold', 0.002)  # 기본 0.2%
        
        # 메트릭 저장소
        self.latency_history = deque(maxlen=self.window_size)
        self.fill_rate_history = deque(maxlen=self.window_size)
        self.slippage_history = deque(maxlen=self.window_size)
        self.cost_history = deque(maxlen=self.window_size)
        
        # 현재 상태
        self.current_latency = 0.0
        self.current_fill_rate = 0.0
        self.current_slippage = 0.0
        self.current_cost = 0.0
        
        self.logger.info("실행 모니터 초기화 완료")
        
    def update_metrics(self, execution_data: Dict[str, Any]) -> None:
        """실행 메트릭 업데이트"""
        try:
            for metric, value in execution_data.items():
                if metric in self.metrics:
                    self.metrics[metric]['current'] = value
                    self._update_average(metric, value)
                    self._update_extremes(metric, value)
                    self._check_threshold(metric)
        except Exception as e:
            self.logger.error(f"메트릭 업데이트 실패: {str(e)}")
            
    def _update_average(self, metric: str, value: float):
        """평균값 업데이트"""
        current_avg = self.metrics[metric]['average']
        if current_avg == 0:
            self.metrics[metric]['average'] = value
        else:
            self.metrics[metric]['average'] = (current_avg + value) / 2

    def _update_extremes(self, metric: str, value: float):
        """극값 업데이트"""
        if metric in ['latency', 'slippage', 'cost']:
            self.metrics[metric]['max'] = max(self.metrics[metric]['max'], value)
        elif metric == 'fill_rate':
            self.metrics[metric]['min'] = min(self.metrics[metric]['min'], value)

    def _check_threshold(self, metric: str):
        """임계값 체크"""
        if metric in ['latency', 'slippage', 'cost']:
            self.metrics[metric]['is_healthy'] = (
                self.metrics[metric]['current'] <= self.metrics[metric]['threshold']
            )
        elif metric == 'fill_rate':
            self.metrics[metric]['is_healthy'] = (
                self.metrics[metric]['current'] >= self.metrics[metric]['threshold']
            )

    def get_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        return self.metrics
            
    def is_healthy(self) -> bool:
        """상태 확인"""
        return all(metric['is_healthy'] for metric in self.metrics.values())
            
    def _calculate_latency(self, execution_data: Dict[str, Any]) -> float:
        """지연시간 계산"""
        try:
            request_time = execution_data.get('request_time')
            response_time = execution_data.get('response_time')
            
            if not request_time or not response_time:
                return 0.0
                
            return (response_time - request_time).total_seconds()
            
        except Exception as e:
            self.logger.error(f"지연시간 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_fill_rate(self, execution_data: Dict[str, Any]) -> float:
        """체결률 계산"""
        try:
            requested_quantity = float(execution_data.get('requested_quantity', 0))
            executed_quantity = float(execution_data.get('executed_quantity', 0))
            
            if requested_quantity == 0:
                return 0.0
                
            return executed_quantity / requested_quantity
            
        except Exception as e:
            self.logger.error(f"체결률 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_slippage(self, execution_data: Dict[str, Any]) -> float:
        """슬리피지 계산"""
        try:
            requested_price = float(execution_data.get('requested_price', 0))
            executed_price = float(execution_data.get('executed_price', 0))
            
            if requested_price == 0:
                return 0.0
                
            return abs(executed_price - requested_price) / requested_price
            
        except Exception as e:
            self.logger.error(f"슬리피지 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_cost(self, execution_data: Dict[str, Any]) -> float:
        """실행 비용 계산"""
        try:
            # 거래 수수료
            fee_rate = self.config.get('fee_rate', 0.001)  # 기본 0.1%
            executed_quantity = float(execution_data.get('executed_quantity', 0))
            executed_price = float(execution_data.get('executed_price', 0))
            
            # 슬리피지 비용
            slippage = self._calculate_slippage(execution_data)
            
            # 총 비용 계산
            fee_cost = executed_quantity * executed_price * fee_rate
            slippage_cost = executed_quantity * executed_price * slippage
            total_cost = fee_cost + slippage_cost
            
            # 정규화된 비용 (거래 금액 대비)
            if executed_quantity * executed_price == 0:
                return 0.0
                
            return total_cost / (executed_quantity * executed_price)
            
        except Exception as e:
            self.logger.error(f"실행 비용 계산 실패: {str(e)}")
            return 0.0 