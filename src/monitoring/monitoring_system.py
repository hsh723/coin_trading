import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..utils.logger import setup_logger
from ..database.database import Database
from .error_handler import ErrorHandler

class MonitoringSystem:
    """모니터링 시스템 클래스"""
    
    def __init__(self):
        """모니터링 시스템 클래스 초기화"""
        self.logger = setup_logger('monitoring_system')
        self.db = Database()
        self.error_handler = ErrorHandler()
        self.metrics = {}
        self.alerts = []
        self.last_check_time = datetime.now()
        
    def start_monitoring(self) -> None:
        """모니터링 시작"""
        try:
            self.logger.info("모니터링 시스템 시작")
            self._initialize_metrics()
            
        except Exception as e:
            self.error_handler.handle_error(e, {'context': '모니터링 시스템 시작'})
            
    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        try:
            self.logger.info("모니터링 시스템 중지")
            self._save_final_metrics()
            
        except Exception as e:
            self.error_handler.handle_error(e, {'context': '모니터링 시스템 중지'})
            
    def update_metrics(self, metric_name: str, value: Any) -> None:
        """
        메트릭 업데이트
        
        Args:
            metric_name (str): 메트릭 이름
            value (Any): 메트릭 값
        """
        try:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = {
                    'values': [],
                    'timestamps': [],
                    'min': float('inf'),
                    'max': float('-inf'),
                    'sum': 0,
                    'count': 0
                }
                
            # 메트릭 업데이트
            self.metrics[metric_name]['values'].append(value)
            self.metrics[metric_name]['timestamps'].append(datetime.now())
            self.metrics[metric_name]['min'] = min(self.metrics[metric_name]['min'], value)
            self.metrics[metric_name]['max'] = max(self.metrics[metric_name]['max'], value)
            self.metrics[metric_name]['sum'] += value
            self.metrics[metric_name]['count'] += 1
            
            # 임계값 확인
            self._check_thresholds(metric_name, value)
            
            # 주기적 저장
            if (datetime.now() - self.last_check_time).seconds >= 300:  # 5분마다
                self._save_metrics()
                self.last_check_time = datetime.now()
                
        except Exception as e:
            self.error_handler.handle_error(e, {
                'context': '메트릭 업데이트',
                'metric_name': metric_name,
                'value': value
            })
            
    def _check_thresholds(self, metric_name: str, value: Any) -> None:
        """
        임계값 확인
        
        Args:
            metric_name (str): 메트릭 이름
            value (Any): 메트릭 값
        """
        try:
            # 임계값 설정 (구현 필요)
            thresholds = {
                'cpu_usage': 80,
                'memory_usage': 90,
                'latency': 1000,
                'error_rate': 0.05
            }
            
            if metric_name in thresholds:
                threshold = thresholds[metric_name]
                if value > threshold:
                    self._create_alert(metric_name, value, threshold)
                    
        except Exception as e:
            self.error_handler.handle_error(e, {
                'context': '임계값 확인',
                'metric_name': metric_name,
                'value': value
            })
            
    def _create_alert(self, metric_name: str, value: Any, threshold: Any) -> None:
        """
        알림 생성
        
        Args:
            metric_name (str): 메트릭 이름
            value (Any): 현재 값
            threshold (Any): 임계값
        """
        try:
            alert = {
                'timestamp': datetime.now(),
                'metric': metric_name,
                'value': value,
                'threshold': threshold,
                'status': 'active'
            }
            
            self.alerts.append(alert)
            self.logger.warning(f"알림 생성: {alert}")
            
            # 알림 전송 (구현 필요)
            # self.notifier.send_notification(alert)
            
        except Exception as e:
            self.error_handler.handle_error(e, {
                'context': '알림 생성',
                'metric_name': metric_name,
                'value': value,
                'threshold': threshold
            })
            
    def _save_metrics(self) -> None:
        """메트릭 저장"""
        try:
            for metric_name, metric_data in self.metrics.items():
                self.db.save_metric({
                    'name': metric_name,
                    'data': metric_data,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            self.error_handler.handle_error(e, {'context': '메트릭 저장'})
            
    def _save_final_metrics(self) -> None:
        """최종 메트릭 저장"""
        try:
            self._save_metrics()
            self.metrics = {}
            
        except Exception as e:
            self.error_handler.handle_error(e, {'context': '최종 메트릭 저장'})
            
    def _initialize_metrics(self) -> None:
        """메트릭 초기화"""
        try:
            self.metrics = {
                'cpu_usage': {'values': [], 'timestamps': [], 'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0},
                'memory_usage': {'values': [], 'timestamps': [], 'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0},
                'latency': {'values': [], 'timestamps': [], 'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0},
                'error_rate': {'values': [], 'timestamps': [], 'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0}
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, {'context': '메트릭 초기화'})
            
    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """
        메트릭 통계 조회
        
        Args:
            metric_name (str): 메트릭 이름
            
        Returns:
            Dict[str, Any]: 메트릭 통계
        """
        try:
            if metric_name in self.metrics:
                metric_data = self.metrics[metric_name]
                return {
                    'min': metric_data['min'],
                    'max': metric_data['max'],
                    'avg': metric_data['sum'] / metric_data['count'] if metric_data['count'] > 0 else 0,
                    'count': metric_data['count']
                }
            return {}
            
        except Exception as e:
            self.error_handler.handle_error(e, {
                'context': '메트릭 통계 조회',
                'metric_name': metric_name
            })
            return {}
            
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        활성 알림 조회
        
        Returns:
            List[Dict[str, Any]]: 활성 알림 목록
        """
        try:
            return [alert for alert in self.alerts if alert['status'] == 'active']
            
        except Exception as e:
            self.error_handler.handle_error(e, {'context': '활성 알림 조회'})
            return []
            
    def resolve_alert(self, alert_id: int) -> None:
        """
        알림 해결
        
        Args:
            alert_id (int): 알림 ID
        """
        try:
            if 0 <= alert_id < len(self.alerts):
                self.alerts[alert_id]['status'] = 'resolved'
                self.logger.info(f"알림 해결: {self.alerts[alert_id]}")
                
        except Exception as e:
            self.error_handler.handle_error(e, {
                'context': '알림 해결',
                'alert_id': alert_id
            }) 