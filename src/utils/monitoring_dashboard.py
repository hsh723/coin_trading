"""
성능 모니터링 대시보드 모듈
"""

import logging
import time
import psutil
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import queue

class MonitoringDashboard:
    """성능 모니터링 대시보드 클래스"""
    
    def __init__(self, db_manager, update_interval: int = 60):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            update_interval (int): 업데이트 간격(초)
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.update_interval = update_interval
        self.metrics = {
            'resource_usage': [],
            'api_response_times': [],
            'data_processing_speeds': []
        }
        self.metrics_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.update_thread = None
        
    def start_monitoring(self):
        """모니터링 시작"""
        try:
            self.stop_event.clear()
            self.update_thread = threading.Thread(
                target=self._update_metrics
            )
            self.update_thread.start()
            
        except Exception as e:
            self.logger.error(f"모니터링 시작 실패: {str(e)}")
            
    def stop_monitoring(self):
        """모니터링 중지"""
        try:
            self.stop_event.set()
            if self.update_thread:
                self.update_thread.join()
                
        except Exception as e:
            self.logger.error(f"모니터링 중지 실패: {str(e)}")
            
    def _update_metrics(self):
        """메트릭 업데이트"""
        while not self.stop_event.is_set():
            try:
                # 리소스 사용량 측정
                resource_usage = self._measure_resource_usage()
                
                # API 응답 시간 측정
                api_response_times = self._measure_api_response_times()
                
                # 데이터 처리 속도 측정
                data_processing_speeds = self._measure_data_processing_speeds()
                
                # 메트릭 저장
                with self.metrics_lock:
                    timestamp = datetime.now()
                    self.metrics['resource_usage'].append({
                        'timestamp': timestamp,
                        **resource_usage
                    })
                    self.metrics['api_response_times'].append({
                        'timestamp': timestamp,
                        **api_response_times
                    })
                    self.metrics['data_processing_speeds'].append({
                        'timestamp': timestamp,
                        **data_processing_speeds
                    })
                    
                # 데이터베이스에 저장
                self._save_metrics_to_db()
                
                # 오래된 데이터 정리
                self._cleanup_old_metrics()
                
            except Exception as e:
                self.logger.error(f"메트릭 업데이트 실패: {str(e)}")
                
            time.sleep(self.update_interval)
            
    def _measure_resource_usage(self) -> Dict:
        """
        리소스 사용량 측정
        
        Returns:
            Dict: 리소스 사용량
        """
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            }
            
        except Exception as e:
            self.logger.error(f"리소스 사용량 측정 실패: {str(e)}")
            return {}
            
    def _measure_api_response_times(self) -> Dict:
        """
        API 응답 시간 측정
        
        Returns:
            Dict: API 응답 시간
        """
        try:
            # API 호출 및 응답 시간 측정
            start_time = time.time()
            # API 호출 코드
            response_time = time.time() - start_time
            
            return {
                'average_response_time': response_time,
                'min_response_time': response_time,
                'max_response_time': response_time
            }
            
        except Exception as e:
            self.logger.error(f"API 응답 시간 측정 실패: {str(e)}")
            return {}
            
    def _measure_data_processing_speeds(self) -> Dict:
        """
        데이터 처리 속도 측정
        
        Returns:
            Dict: 데이터 처리 속도
        """
        try:
            # 데이터 처리 속도 측정
            start_time = time.time()
            # 데이터 처리 코드
            processing_time = time.time() - start_time
            
            return {
                'processing_speed': 1 / processing_time if processing_time > 0 else 0,
                'data_volume': 0  # 처리된 데이터 양
            }
            
        except Exception as e:
            self.logger.error(f"데이터 처리 속도 측정 실패: {str(e)}")
            return {}
            
    def _save_metrics_to_db(self):
        """메트릭을 데이터베이스에 저장"""
        try:
            with self.metrics_lock:
                metrics_data = {
                    'timestamp': datetime.now(),
                    'resource_usage': self.metrics['resource_usage'][-1],
                    'api_response_times': self.metrics['api_response_times'][-1],
                    'data_processing_speeds': self.metrics['data_processing_speeds'][-1]
                }
                
            self.db_manager.save_monitoring_metrics(metrics_data)
            
        except Exception as e:
            self.logger.error(f"메트릭 저장 실패: {str(e)}")
            
    def _cleanup_old_metrics(self):
        """오래된 메트릭 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            with self.metrics_lock:
                for metric_type in self.metrics:
                    self.metrics[metric_type] = [
                        m for m in self.metrics[metric_type]
                        if m['timestamp'] > cutoff_time
                    ]
                    
        except Exception as e:
            self.logger.error(f"메트릭 정리 실패: {str(e)}")
            
    def get_resource_usage_chart(self) -> go.Figure:
        """
        리소스 사용량 차트 생성
        
        Returns:
            go.Figure: 리소스 사용량 차트
        """
        try:
            with self.metrics_lock:
                df = pd.DataFrame(self.metrics['resource_usage'])
                
            fig = go.Figure()
            
            # CPU 사용량
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['cpu_percent'],
                name='CPU 사용량',
                line=dict(color='blue')
            ))
            
            # 메모리 사용량
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['memory_percent'],
                name='메모리 사용량',
                line=dict(color='red')
            ))
            
            # 디스크 사용량
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['disk_percent'],
                name='디스크 사용량',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title='리소스 사용량',
                xaxis_title='시간',
                yaxis_title='사용량 (%)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"리소스 사용량 차트 생성 실패: {str(e)}")
            return go.Figure()
            
    def get_api_performance_chart(self) -> go.Figure:
        """
        API 성능 차트 생성
        
        Returns:
            go.Figure: API 성능 차트
        """
        try:
            with self.metrics_lock:
                df = pd.DataFrame(self.metrics['api_response_times'])
                
            fig = go.Figure()
            
            # 평균 응답 시간
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['average_response_time'],
                name='평균 응답 시간',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='API 응답 시간',
                xaxis_title='시간',
                yaxis_title='응답 시간 (초)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"API 성능 차트 생성 실패: {str(e)}")
            return go.Figure()
            
    def get_data_processing_chart(self) -> go.Figure:
        """
        데이터 처리 속도 차트 생성
        
        Returns:
            go.Figure: 데이터 처리 속도 차트
        """
        try:
            with self.metrics_lock:
                df = pd.DataFrame(self.metrics['data_processing_speeds'])
                
            fig = go.Figure()
            
            # 처리 속도
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['processing_speed'],
                name='처리 속도',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='데이터 처리 속도',
                xaxis_title='시간',
                yaxis_title='처리 속도 (ops/sec)',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"데이터 처리 속도 차트 생성 실패: {str(e)}")
            return go.Figure()
            
    def get_system_status(self) -> Dict:
        """
        시스템 상태 조회
        
        Returns:
            Dict: 시스템 상태
        """
        try:
            with self.metrics_lock:
                latest_metrics = {
                    'resource_usage': self.metrics['resource_usage'][-1] if self.metrics['resource_usage'] else {},
                    'api_response_times': self.metrics['api_response_times'][-1] if self.metrics['api_response_times'] else {},
                    'data_processing_speeds': self.metrics['data_processing_speeds'][-1] if self.metrics['data_processing_speeds'] else {}
                }
                
            return {
                'status': 'normal' if self._check_system_health(latest_metrics) else 'warning',
                'timestamp': datetime.now(),
                'metrics': latest_metrics
            }
            
        except Exception as e:
            self.logger.error(f"시스템 상태 조회 실패: {str(e)}")
            return {'status': 'error'}
            
    def _check_system_health(self, metrics: Dict) -> bool:
        """
        시스템 건강 상태 확인
        
        Args:
            metrics (Dict): 시스템 메트릭
            
        Returns:
            bool: 시스템 건강 상태
        """
        try:
            # 리소스 사용량 확인
            resource_usage = metrics.get('resource_usage', {})
            if (resource_usage.get('cpu_percent', 0) > 90 or
                resource_usage.get('memory_percent', 0) > 90 or
                resource_usage.get('disk_percent', 0) > 90):
                return False
                
            # API 응답 시간 확인
            api_times = metrics.get('api_response_times', {})
            if api_times.get('average_response_time', 0) > 5.0:  # 5초 이상
                return False
                
            # 데이터 처리 속도 확인
            processing_speeds = metrics.get('data_processing_speeds', {})
            if processing_speeds.get('processing_speed', 0) < 1.0:  # 1 ops/sec 미만
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"시스템 건강 상태 확인 실패: {str(e)}")
            return False 