import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
from ..utils.logger import setup_logger

logger = setup_logger()

class PerformanceTracker:
    """
    시스템 성능 메트릭을 추적하는 도구
    
    Attributes:
        history_size (int): 저장할 히스토리 크기
        check_interval (float): 체크 간격 (초)
        metrics (Dict): 추적할 메트릭 정의
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        check_interval: float = 1.0,
        metrics: Optional[Dict] = None
    ):
        self.history_size = history_size
        self.check_interval = check_interval
        
        # 기본 메트릭 정의
        self.metrics = metrics or {
            'cpu_usage': {'type': 'gauge', 'unit': '%'},
            'memory_usage': {'type': 'gauge', 'unit': '%'},
            'disk_usage': {'type': 'gauge', 'unit': '%'},
            'network_io': {'type': 'counter', 'unit': 'bytes'},
            'response_time': {'type': 'gauge', 'unit': 'ms'},
            'error_rate': {'type': 'gauge', 'unit': '%'},
            'trading_volume': {'type': 'counter', 'unit': 'orders'},
            'profit_loss': {'type': 'gauge', 'unit': 'USD'}
        }
        
        # 메트릭 히스토리
        self.metric_history = {
            metric: deque(maxlen=history_size)
            for metric in self.metrics
        }
        
        # 성능 임계값
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'disk_usage': 80.0,
            'error_rate': 5.0
        }
        
        # 알림 상태
        self.alerts = []
        self.last_alert_time = {}
    
    def start_tracking(self):
        """성능 추적을 시작합니다."""
        try:
            while True:
                try:
                    self._collect_metrics()
                    self._check_thresholds()
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error tracking performance: {str(e)}")
                    time.sleep(5)  # 5초 대기 후 재시도
                    
        except KeyboardInterrupt:
            logger.info("Performance tracking stopped")
    
    def _collect_metrics(self):
        """메트릭을 수집합니다."""
        timestamp = datetime.now().isoformat()
        
        for metric, config in self.metrics.items():
            try:
                value = self._get_metric_value(metric)
                if value is not None:
                    self._update_metric(metric, timestamp, value)
                    
            except Exception as e:
                logger.error(f"Error collecting metric {metric}: {str(e)}")
    
    def _get_metric_value(self, metric: str) -> Optional[float]:
        """
        메트릭 값을 가져옵니다.
        
        Args:
            metric (str): 메트릭 이름
            
        Returns:
            Optional[float]: 메트릭 값
        """
        # 여기에 실제 메트릭 수집 로직 구현
        # 예: psutil, API 호출 등
        return None
    
    def _update_metric(self, metric: str, timestamp: str, value: float):
        """
        메트릭을 업데이트합니다.
        
        Args:
            metric (str): 메트릭 이름
            timestamp (str): 타임스탬프
            value (float): 메트릭 값
        """
        metric_data = {
            'timestamp': timestamp,
            'value': value
        }
        
        self.metric_history[metric].append(metric_data)
    
    def _check_thresholds(self):
        """임계값을 확인합니다."""
        current_time = datetime.now()
        
        for metric, threshold in self.thresholds.items():
            if metric in self.metric_history:
                latest_value = self._get_latest_value(metric)
                if latest_value is not None and latest_value >= threshold:
                    self._handle_threshold_breach(metric, latest_value, current_time)
    
    def _get_latest_value(self, metric: str) -> Optional[float]:
        """
        최신 메트릭 값을 가져옵니다.
        
        Args:
            metric (str): 메트릭 이름
            
        Returns:
            Optional[float]: 최신 메트릭 값
        """
        if self.metric_history[metric]:
            return self.metric_history[metric][-1]['value']
        return None
    
    def _handle_threshold_breach(self, metric: str, value: float, current_time: datetime):
        """
        임계값 초과를 처리합니다.
        
        Args:
            metric (str): 메트릭 이름
            value (float): 현재 값
            current_time (datetime): 현재 시간
        """
        # 마지막 알림 시간 확인
        last_alert = self.last_alert_time.get(metric)
        if last_alert:
            time_since_last_alert = (current_time - last_alert).total_seconds()
            if time_since_last_alert < 300:  # 5분 이내 재알림 방지
                return
        
        # 알림 생성
        alert = {
            'timestamp': current_time.isoformat(),
            'metric': metric,
            'value': value,
            'threshold': self.thresholds[metric],
            'message': f"{metric} threshold exceeded: {value}%"
        }
        
        self.alerts.append(alert)
        self.last_alert_time[metric] = current_time
        
        logger.warning(f"Alert: {alert['message']}")
        
        # 여기에 알림 전송 로직 추가 (이메일, 텔레그램 등)
    
    def get_performance_report(self) -> Dict:
        """
        성능 리포트를 생성합니다.
        
        Returns:
            Dict: 성능 리포트
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                metric: list(history)
                for metric, history in self.metric_history.items()
            },
            'active_alerts': self.alerts
        }
    
    def get_performance_stats(self, period: str = '1h') -> Dict:
        """
        성능 통계를 계산합니다.
        
        Args:
            period (str): 통계 기간 ('1h', '24h', '7d')
            
        Returns:
            Dict: 성능 통계
        """
        end_time = datetime.now()
        
        if period == '1h':
            start_time = end_time - timedelta(hours=1)
        elif period == '24h':
            start_time = end_time - timedelta(days=1)
        elif period == '7d':
            start_time = end_time - timedelta(days=7)
        else:
            raise ValueError(f"Invalid period: {period}")
        
        stats = {}
        
        for metric, history in self.metric_history.items():
            metric_config = self.metrics[metric]
            values = [
                data['value']
                for data in history
                if start_time <= datetime.fromisoformat(data['timestamp']) <= end_time
            ]
            
            if values:
                stats[metric] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        return stats
    
    def clear_alerts(self):
        """활성 알림을 초기화합니다."""
        self.alerts = []
        self.last_alert_time = {}
    
    def add_metric(self, name: str, metric_type: str, unit: str):
        """
        새로운 메트릭을 추가합니다.
        
        Args:
            name (str): 메트릭 이름
            metric_type (str): 메트릭 타입 ('gauge' 또는 'counter')
            unit (str): 단위
        """
        self.metrics[name] = {
            'type': metric_type,
            'unit': unit
        }
        self.metric_history[name] = deque(maxlen=self.history_size)
    
    def remove_metric(self, name: str):
        """
        메트릭을 제거합니다.
        
        Args:
            name (str): 메트릭 이름
        """
        if name in self.metrics:
            del self.metrics[name]
            del self.metric_history[name]
    
    def set_threshold(self, metric: str, threshold: float):
        """
        임계값을 설정합니다.
        
        Args:
            metric (str): 메트릭 이름
            threshold (float): 새로운 임계값
        """
        self.thresholds[metric] = threshold 