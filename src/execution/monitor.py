"""
실행 시스템 모니터링 모듈
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from src.execution.logger import ExecutionLogger
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class ExecutionMonitor:
    """실행 시스템 모니터"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        모니터 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        self.logger = ExecutionLogger(config)
        
        # 모니터링 설정
        self.monitoring_interval = config['monitoring'].get('interval', 60)  # 초
        self.alert_thresholds = config['monitoring'].get('alert_thresholds', {
            'latency': 1000,  # ms
            'error_rate': 0.05,  # 5%
            'fill_rate': 0.95,  # 95%
            'slippage': 0.001  # 0.1%
        })
        
        # 성능 메트릭
        self.metrics = {
            'latency': [],
            'error_rate': [],
            'fill_rate': [],
            'slippage': [],
            'volume': [],
            'active_orders': 0
        }
        
        # 이상 탐지 설정
        self.anomaly_window = 60  # 1시간
        self.z_score_threshold = 3.0
        
    async def initialize(self):
        """모니터 초기화"""
        try:
            # 로거 초기화
            await self.logger.initialize()
            
            # 모니터링 시작
            asyncio.create_task(self._monitoring_loop())
            
            logger.info("실행 모니터 초기화 완료")
            
        except Exception as e:
            logger.error(f"모니터 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            await self.logger.close()
            logger.info("실행 모니터 종료")
        except Exception as e:
            logger.error(f"모니터 종료 실패: {str(e)}")
            
    async def _monitoring_loop(self):
        """모니터링 루프"""
        try:
            while True:
                # 성능 메트릭 수집
                await self._collect_metrics()
                
                # 이상 탐지
                await self._detect_anomalies()
                
                # 성능 분석
                await self._analyze_performance()
                
                # 대기
                await asyncio.sleep(self.monitoring_interval)
                
        except Exception as e:
            logger.error(f"모니터링 루프 실행 중 오류 발생: {str(e)}")
            
    async def _collect_metrics(self):
        """성능 메트릭 수집"""
        try:
            # 현재 시간
            now = datetime.now()
            start_time = now - timedelta(seconds=self.monitoring_interval)
            
            # 실행 로그 조회
            execution_logs = await self.logger.get_execution_logs(
                start_time=start_time,
                end_time=now
            )
            
            if len(execution_logs) > 0:
                # 지연 시간 계산
                latencies = execution_logs['latency'].values
                self.metrics['latency'].append(np.mean(latencies))
                
                # 체결률 계산
                fill_rates = execution_logs['fill_rate'].values
                self.metrics['fill_rate'].append(np.mean(fill_rates))
                
                # 슬리피지 계산
                slippages = execution_logs['slippage'].values
                self.metrics['slippage'].append(np.mean(slippages))
                
                # 거래량 계산
                volumes = execution_logs['volume'].values
                self.metrics['volume'].append(np.sum(volumes))
                
            # 오류 로그 조회
            error_logs = await self.logger.get_error_logs(
                start_time=start_time,
                end_time=now
            )
            
            # 오류율 계산
            total_executions = len(execution_logs)
            total_errors = len(error_logs)
            if total_executions > 0:
                error_rate = total_errors / total_executions
            else:
                error_rate = 0.0
            self.metrics['error_rate'].append(error_rate)
            
            # 메트릭 기록
            await self.logger.log_performance({
                'latency': self.metrics['latency'][-1],
                'error_rate': self.metrics['error_rate'][-1],
                'fill_rate': self.metrics['fill_rate'][-1],
                'slippage': self.metrics['slippage'][-1],
                'volume': self.metrics['volume'][-1]
            })
            
        except Exception as e:
            logger.error(f"메트릭 수집 실패: {str(e)}")
            
    async def _detect_anomalies(self):
        """이상 탐지"""
        try:
            for metric_name, values in self.metrics.items():
                if isinstance(values, list) and len(values) >= self.anomaly_window:
                    # 최근 데이터
                    recent_values = values[-self.anomaly_window:]
                    
                    # Z-score 계산
                    mean = np.mean(recent_values)
                    std = np.std(recent_values)
                    z_score = abs((recent_values[-1] - mean) / std)
                    
                    # 이상 탐지
                    if z_score > self.z_score_threshold:
                        await self._handle_anomaly(metric_name, recent_values[-1], z_score)
                        
        except Exception as e:
            logger.error(f"이상 탐지 실패: {str(e)}")
            
    async def _handle_anomaly(
        self,
        metric_name: str,
        value: float,
        z_score: float
    ):
        """
        이상 처리
        
        Args:
            metric_name (str): 메트릭 이름
            value (float): 현재 값
            z_score (float): Z-score
        """
        try:
            # 임계값 확인
            threshold = self.alert_thresholds.get(metric_name)
            if threshold and value > threshold:
                # 오류 로그 기록
                await self.logger.log_error({
                    'code': 'ANOMALY',
                    'message': f'이상 탐지: {metric_name}',
                    'details': {
                        'value': value,
                        'z_score': z_score,
                        'threshold': threshold
                    }
                })
                
        except Exception as e:
            logger.error(f"이상 처리 실패: {str(e)}")
            
    async def _analyze_performance(self):
        """성능 분석"""
        try:
            # 성능 메트릭 요약
            summary = {}
            for metric_name, values in self.metrics.items():
                if isinstance(values, list) and len(values) > 0:
                    summary[metric_name] = {
                        'current': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                    
            # 성능 로그 기록
            await self.logger.log_performance({
                'type': 'summary',
                'metrics': summary
            })
            
        except Exception as e:
            logger.error(f"성능 분석 실패: {str(e)}")
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        메트릭 요약 조회
        
        Returns:
            Dict[str, Any]: 메트릭 요약
        """
        summary = {}
        for metric_name, values in self.metrics.items():
            if isinstance(values, list) and len(values) > 0:
                summary[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return summary
        
    async def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        메트릭 이력 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 메트릭 이력
        """
        try:
            # 성능 로그 조회
            logs = await self.logger.get_performance_logs(
                start_time=start_time,
                end_time=end_time
            )
            
            # 데이터 변환
            metrics_data = []
            for _, row in logs.iterrows():
                metrics_data.append({
                    'timestamp': row['timestamp'],
                    **{
                        k: v for k, v in row.items()
                        if k != 'timestamp' and not isinstance(v, dict)
                    }
                })
                
            return pd.DataFrame(metrics_data)
            
        except Exception as e:
            logger.error(f"메트릭 이력 조회 실패: {str(e)}")
            return pd.DataFrame()
            
    async def get_anomalies(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        이상 탐지 이력 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 이상 탐지 이력
        """
        try:
            # 오류 로그 조회
            logs = await self.logger.get_error_logs(
                start_time=start_time,
                end_time=end_time
            )
            
            # 이상 탐지 필터링
            anomalies = logs[logs['code'] == 'ANOMALY']
            
            return anomalies
            
        except Exception as e:
            logger.error(f"이상 탐지 이력 조회 실패: {str(e)}")
            return pd.DataFrame() 