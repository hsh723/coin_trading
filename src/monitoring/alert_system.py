import time
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import deque
from ..utils.logger import setup_logger

logger = setup_logger()

class AlertSystem:
    """
    임계값 기반 알림 시스템
    
    Attributes:
        history_size (int): 저장할 히스토리 크기
        check_interval (float): 체크 간격 (초)
        alerts (Dict): 알림 정의
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        check_interval: float = 1.0,
        alerts: Optional[Dict] = None
    ):
        self.history_size = history_size
        self.check_interval = check_interval
        
        # 기본 알림 정의
        self.alerts = alerts or {
            'high_cpu': {
                'condition': lambda x: x > 80,
                'message': 'CPU 사용량이 80%를 초과했습니다',
                'severity': 'warning',
                'cooldown': 300  # 5분
            },
            'high_memory': {
                'condition': lambda x: x > 80,
                'message': '메모리 사용량이 80%를 초과했습니다',
                'severity': 'warning',
                'cooldown': 300
            },
            'high_error_rate': {
                'condition': lambda x: x > 5,
                'message': '오류율이 5%를 초과했습니다',
                'severity': 'error',
                'cooldown': 300
            },
            'low_profit': {
                'condition': lambda x: x < -1000,
                'message': '손실이 $1,000를 초과했습니다',
                'severity': 'error',
                'cooldown': 300
            }
        }
        
        # 알림 히스토리
        self.alert_history = deque(maxlen=history_size)
        
        # 알림 상태
        self.active_alerts = {}
        self.last_alert_time = {}
        
        # 알림 핸들러
        self.handlers = {
            'warning': self._handle_warning,
            'error': self._handle_error,
            'critical': self._handle_critical
        }
    
    def start_monitoring(self):
        """알림 모니터링을 시작합니다."""
        try:
            while True:
                try:
                    self._check_alerts()
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error monitoring alerts: {str(e)}")
                    time.sleep(5)  # 5초 대기 후 재시도
                    
        except KeyboardInterrupt:
            logger.info("Alert monitoring stopped")
    
    def _check_alerts(self):
        """알림 조건을 확인합니다."""
        current_time = datetime.now()
        
        for alert_name, alert_config in self.alerts.items():
            try:
                # 알림 쿨다운 확인
                last_alert = self.last_alert_time.get(alert_name)
                if last_alert:
                    time_since_last_alert = (current_time - last_alert).total_seconds()
                    if time_since_last_alert < alert_config['cooldown']:
                        continue
                
                # 알림 조건 확인
                if alert_config['condition'](self._get_alert_value(alert_name)):
                    self._trigger_alert(alert_name, alert_config, current_time)
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert_name}: {str(e)}")
    
    def _get_alert_value(self, alert_name: str) -> Optional[float]:
        """
        알림 값을 가져옵니다.
        
        Args:
            alert_name (str): 알림 이름
            
        Returns:
            Optional[float]: 알림 값
        """
        # 여기에 실제 값 수집 로직 구현
        # 예: 메트릭 수집, API 호출 등
        return None
    
    def _trigger_alert(self, alert_name: str, alert_config: Dict, current_time: datetime):
        """
        알림을 트리거합니다.
        
        Args:
            alert_name (str): 알림 이름
            alert_config (Dict): 알림 설정
            current_time (datetime): 현재 시간
        """
        alert = {
            'timestamp': current_time.isoformat(),
            'name': alert_name,
            'message': alert_config['message'],
            'severity': alert_config['severity'],
            'value': self._get_alert_value(alert_name)
        }
        
        # 알림 히스토리 추가
        self.alert_history.append(alert)
        
        # 알림 상태 업데이트
        self.active_alerts[alert_name] = alert
        self.last_alert_time[alert_name] = current_time
        
        # 알림 핸들러 호출
        handler = self.handlers.get(alert_config['severity'])
        if handler:
            handler(alert)
        
        logger.warning(f"Alert triggered: {alert['message']}")
    
    def _handle_warning(self, alert: Dict):
        """경고 알림을 처리합니다."""
        # 여기에 경고 알림 처리 로직 구현
        # 예: 이메일 전송, 로깅 등
        pass
    
    def _handle_error(self, alert: Dict):
        """오류 알림을 처리합니다."""
        # 여기에 오류 알림 처리 로직 구현
        # 예: 이메일 전송, 로깅 등
        pass
    
    def _handle_critical(self, alert: Dict):
        """심각한 알림을 처리합니다."""
        # 여기에 심각한 알림 처리 로직 구현
        # 예: 이메일 전송, 로깅, 시스템 종료 등
        pass
    
    def get_alert_report(self) -> Dict:
        """
        알림 리포트를 생성합니다.
        
        Returns:
            Dict: 알림 리포트
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'active_alerts': self.active_alerts,
            'alert_history': list(self.alert_history)
        }
    
    def get_alert_stats(self) -> Dict:
        """
        알림 통계를 계산합니다.
        
        Returns:
            Dict: 알림 통계
        """
        alert_history = list(self.alert_history)
        
        # 심각도별 분포
        severity_distribution = {}
        for alert in alert_history:
            severity = alert['severity']
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        # 알림 유형별 분포
        type_distribution = {}
        for alert in alert_history:
            alert_type = alert['name']
            type_distribution[alert_type] = type_distribution.get(alert_type, 0) + 1
        
        return {
            'total_alerts': len(alert_history),
            'active_alerts': len(self.active_alerts),
            'severity_distribution': severity_distribution,
            'type_distribution': type_distribution
        }
    
    def clear_alerts(self):
        """활성 알림을 초기화합니다."""
        self.active_alerts = {}
        self.last_alert_time = {}
    
    def add_alert(
        self,
        name: str,
        condition: Callable,
        message: str,
        severity: str = 'warning',
        cooldown: int = 300
    ):
        """
        새로운 알림을 추가합니다.
        
        Args:
            name (str): 알림 이름
            condition (Callable): 알림 조건 함수
            message (str): 알림 메시지
            severity (str): 심각도 ('warning', 'error', 'critical')
            cooldown (int): 쿨다운 시간 (초)
        """
        self.alerts[name] = {
            'condition': condition,
            'message': message,
            'severity': severity,
            'cooldown': cooldown
        }
    
    def remove_alert(self, name: str):
        """
        알림을 제거합니다.
        
        Args:
            name (str): 알림 이름
        """
        if name in self.alerts:
            del self.alerts[name]
            if name in self.active_alerts:
                del self.active_alerts[name]
            if name in self.last_alert_time:
                del self.last_alert_time[name]
    
    def set_cooldown(self, name: str, cooldown: int):
        """
        알림 쿨다운을 설정합니다.
        
        Args:
            name (str): 알림 이름
            cooldown (int): 새로운 쿨다운 시간 (초)
        """
        if name in self.alerts:
            self.alerts[name]['cooldown'] = cooldown
    
    def add_handler(self, severity: str, handler: Callable):
        """
        알림 핸들러를 추가합니다.
        
        Args:
            severity (str): 심각도
            handler (Callable): 핸들러 함수
        """
        self.handlers[severity] = handler 