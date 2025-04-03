import re
import time
from typing import Dict, List, Optional, Pattern
from datetime import datetime
from collections import deque
from ..utils.logger import setup_logger

logger = setup_logger()

class ErrorDetector:
    """
    시스템 오류를 감지하고 분석하는 도구
    
    Attributes:
        log_file (str): 로그 파일 경로
        error_patterns (Dict): 오류 패턴 정의
        alert_threshold (int): 알림 임계값
        history_size (int): 저장할 히스토리 크기
    """
    
    def __init__(
        self,
        log_file: str = 'logs/trading.log',
        error_patterns: Optional[Dict] = None,
        alert_threshold: int = 5,
        history_size: int = 1000
    ):
        self.log_file = log_file
        self.error_patterns = error_patterns or {
            'connection_error': r'Connection refused|Connection reset|Timeout',
            'api_error': r'API Error|Rate limit exceeded|Invalid API key',
            'data_error': r'Data validation failed|Missing data|Invalid data format',
            'system_error': r'System error|Runtime error|Memory error',
            'trading_error': r'Trading error|Order failed|Position error'
        }
        
        self.alert_threshold = alert_threshold
        self.history_size = history_size
        
        # 오류 히스토리
        self.error_history = deque(maxlen=history_size)
        self.error_counts = {pattern: 0 for pattern in self.error_patterns}
        
        # 알림 상태
        self.alerts = []
        self.last_alert_time = {}
        
        # 패턴 컴파일
        self.compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.error_patterns.items()
        }
    
    def start_monitoring(self):
        """로그 파일 모니터링을 시작합니다."""
        try:
            last_position = 0
            
            while True:
                try:
                    with open(self.log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        
                        if new_lines:
                            self._process_lines(new_lines)
                            last_position = f.tell()
                    
                    time.sleep(1)  # 1초 대기
                    
                except FileNotFoundError:
                    logger.error(f"Log file not found: {self.log_file}")
                    time.sleep(5)  # 5초 대기 후 재시도
                    
                except Exception as e:
                    logger.error(f"Error monitoring log file: {str(e)}")
                    time.sleep(5)  # 5초 대기 후 재시도
                    
        except KeyboardInterrupt:
            logger.info("Error monitoring stopped")
    
    def _process_lines(self, lines: List[str]):
        """로그 라인을 처리합니다."""
        for line in lines:
            self._check_error_patterns(line)
    
    def _check_error_patterns(self, line: str):
        """오류 패턴을 확인합니다."""
        for error_type, pattern in self.compiled_patterns.items():
            if pattern.search(line):
                self._handle_error(error_type, line)
    
    def _handle_error(self, error_type: str, line: str):
        """오류를 처리합니다."""
        timestamp = datetime.now().isoformat()
        error_info = {
            'timestamp': timestamp,
            'type': error_type,
            'message': line.strip(),
            'count': self.error_counts[error_type] + 1
        }
        
        # 오류 카운트 업데이트
        self.error_counts[error_type] += 1
        
        # 오류 히스토리 추가
        self.error_history.append(error_info)
        
        # 알림 체크
        self._check_alerts(error_type)
        
        # 로깅
        logger.warning(f"Error detected: {error_type} - {line.strip()}")
    
    def _check_alerts(self, error_type: str):
        """알림 조건을 확인합니다."""
        current_time = datetime.now()
        
        # 마지막 알림 시간 확인
        last_alert = self.last_alert_time.get(error_type)
        if last_alert:
            time_since_last_alert = (current_time - last_alert).total_seconds()
            if time_since_last_alert < 300:  # 5분 이내 재알림 방지
                return
        
        # 임계값 체크
        if self.error_counts[error_type] >= self.alert_threshold:
            self._send_alert(error_type)
            self.last_alert_time[error_type] = current_time
    
    def _send_alert(self, error_type: str):
        """알림을 전송합니다."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'count': self.error_counts[error_type],
            'message': f"Error threshold exceeded for {error_type}"
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: {alert['message']}")
        
        # 여기에 알림 전송 로직 추가 (이메일, 텔레그램 등)
    
    def get_error_report(self) -> Dict:
        """
        오류 리포트를 생성합니다.
        
        Returns:
            Dict: 오류 리포트
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'error_counts': self.error_counts,
            'recent_errors': list(self.error_history),
            'active_alerts': self.alerts
        }
    
    def get_error_stats(self) -> Dict:
        """
        오류 통계를 계산합니다.
        
        Returns:
            Dict: 오류 통계
        """
        error_history = list(self.error_history)
        
        # 시간대별 오류 분포
        hourly_distribution = {}
        for error in error_history:
            hour = datetime.fromisoformat(error['timestamp']).hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        # 오류 유형별 분포
        type_distribution = {}
        for error in error_history:
            error_type = error['type']
            type_distribution[error_type] = type_distribution.get(error_type, 0) + 1
        
        return {
            'total_errors': len(error_history),
            'hourly_distribution': hourly_distribution,
            'type_distribution': type_distribution,
            'error_counts': self.error_counts
        }
    
    def clear_alerts(self):
        """활성 알림을 초기화합니다."""
        self.alerts = []
        self.last_alert_time = {}
        self.error_counts = {pattern: 0 for pattern in self.error_patterns}
    
    def add_error_pattern(self, name: str, pattern: str):
        """
        새로운 오류 패턴을 추가합니다.
        
        Args:
            name (str): 패턴 이름
            pattern (str): 정규식 패턴
        """
        self.error_patterns[name] = pattern
        self.compiled_patterns[name] = re.compile(pattern)
        self.error_counts[name] = 0
    
    def remove_error_pattern(self, name: str):
        """
        오류 패턴을 제거합니다.
        
        Args:
            name (str): 패턴 이름
        """
        if name in self.error_patterns:
            del self.error_patterns[name]
            del self.compiled_patterns[name]
            del self.error_counts[name]
    
    def set_alert_threshold(self, threshold: int):
        """
        알림 임계값을 설정합니다.
        
        Args:
            threshold (int): 새로운 임계값
        """
        self.alert_threshold = threshold 