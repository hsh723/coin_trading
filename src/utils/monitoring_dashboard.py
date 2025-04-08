"""
모니터링 대시보드 모듈
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psutil
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import telegram
import os

class MonitoringDashboard:
    """모니터링 대시보드 클래스"""
    
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
        self.is_running = False
        self.monitoring_thread = None
        self.alert_thresholds = {
            'cpu': 80,
            'memory': 80,
            'disk': 80,
            'api_response': 5.0,
            'processing_speed': 1000
        }
        self.alert_channels = {
            'email': False,
            'telegram': False
        }
        self.alert_recipients = {
            'email': [],
            'telegram': []
        }
        
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.is_running:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.start()
            self.logger.info("모니터링 시작")
            
    def stop_monitoring(self):
        """모니터링 중지"""
        if self.is_running:
            self.is_running = False
            if self.monitoring_thread:
                self.monitoring_thread.join()
            self.logger.info("모니터링 중지")
            
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                # 리소스 사용량 측정
                resource_usage = self._measure_resource_usage()
                
                # API 응답 시간 측정
                api_performance = self._measure_api_performance()
                
                # 데이터 처리 속도 측정
                processing_speed = self._measure_processing_speed()
                
                # 메트릭 저장
                self._save_metrics({
                    'timestamp': datetime.now(),
                    'resource_usage': resource_usage,
                    'api_performance': api_performance,
                    'processing_speed': processing_speed
                })
                
                # 알림 체크
                self._check_alerts(
                    resource_usage,
                    api_performance,
                    processing_speed
                )
                
                # 오래된 메트릭 정리
                self._cleanup_old_metrics()
                
            except Exception as e:
                self.logger.error(f"모니터링 중 오류 발생: {str(e)}")
                
            time.sleep(self.update_interval)
            
    def _measure_resource_usage(self) -> Dict:
        """
        리소스 사용량 측정
        
        Returns:
            Dict: 리소스 사용량
        """
        try:
            return {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent,
                'network': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            }
            
        except Exception as e:
            self.logger.error(f"리소스 사용량 측정 실패: {str(e)}")
            return {}
            
    def _measure_api_performance(self) -> Dict:
        """
        API 성능 측정
        
        Returns:
            Dict: API 성능 메트릭
        """
        try:
            start_time = time.time()
            response = requests.get('https://api.binance.com/api/v3/ping')
            response_time = (time.time() - start_time) * 1000
            
            return {
                'response_time': response_time,
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
            
        except Exception as e:
            self.logger.error(f"API 성능 측정 실패: {str(e)}")
            return {}
            
    def _measure_processing_speed(self) -> Dict:
        """
        데이터 처리 속도 측정
        
        Returns:
            Dict: 처리 속도 메트릭
        """
        try:
            # 테스트 데이터 생성
            test_data = pd.DataFrame({
                'price': [100 + i for i in range(1000)],
                'volume': [1000 + i for i in range(1000)]
            })
            
            # 처리 시간 측정
            start_time = time.time()
            test_data['sma'] = test_data['price'].rolling(window=20).mean()
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'processing_time': processing_time,
                'data_size': len(test_data)
            }
            
        except Exception as e:
            self.logger.error(f"처리 속도 측정 실패: {str(e)}")
            return {}
            
    def _save_metrics(self, metrics: Dict):
        """
        메트릭 저장
        
        Args:
            metrics (Dict): 저장할 메트릭
        """
        try:
            self.db_manager.save_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"메트릭 저장 실패: {str(e)}")
            
    def _cleanup_old_metrics(self, days: int = 30):
        """
        오래된 메트릭 정리
        
        Args:
            days (int): 보관 기간(일)
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            self.db_manager.cleanup_old_metrics(cutoff_date)
            
        except Exception as e:
            self.logger.error(f"메트릭 정리 실패: {str(e)}")
            
    def _check_alerts(self, resource_usage: Dict, api_performance: Dict, processing_speed: Dict):
        """
        알림 체크
        
        Args:
            resource_usage (Dict): 리소스 사용량
            api_performance (Dict): API 성능
            processing_speed (Dict): 처리 속도
        """
        try:
            alerts = []
            
            # 리소스 사용량 알림
            for resource, usage in resource_usage.items():
                if usage > self.alert_thresholds.get(resource, 0):
                    alerts.append(f"{resource} 사용량이 {usage}%로 임계값을 초과했습니다")
                    
            # API 성능 알림
            if api_performance.get('response_time', 0) > self.alert_thresholds['api_response']:
                alerts.append(f"API 응답 시간이 {api_performance['response_time']}ms로 임계값을 초과했습니다")
                
            # 처리 속도 알림
            if processing_speed.get('processing_time', 0) > self.alert_thresholds['processing_speed']:
                alerts.append(f"데이터 처리 시간이 {processing_speed['processing_time']}ms로 임계값을 초과했습니다")
                
            # 알림 전송
            if alerts:
                self._send_alerts(alerts)
                
        except Exception as e:
            self.logger.error(f"알림 체크 실패: {str(e)}")
            
    def _send_alerts(self, alerts: List[str]):
        """
        알림 전송
        
        Args:
            alerts (List[str]): 알림 메시지 목록
        """
        try:
            message = "\n".join(alerts)
            
            # 이메일 알림
            if self.alert_channels['email']:
                self._send_email_alert(message)
                
            # 텔레그램 알림
            if self.alert_channels['telegram']:
                self._send_telegram_alert(message)
                
        except Exception as e:
            self.logger.error(f"알림 전송 실패: {str(e)}")
            
    def _send_email_alert(self, message: str):
        """
        이메일 알림 전송
        
        Args:
            message (str): 알림 메시지
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = os.getenv('SMTP_USER')
            msg['To'] = ', '.join(self.alert_recipients['email'])
            msg['Subject'] = "시스템 알림"
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(os.getenv('SMTP_HOST'), int(os.getenv('SMTP_PORT'))) as server:
                server.starttls()
                server.login(os.getenv('SMTP_USER'), os.getenv('SMTP_PASSWORD'))
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"이메일 알림 전송 실패: {str(e)}")
            
    def _send_telegram_alert(self, message: str):
        """
        텔레그램 알림 전송
        
        Args:
            message (str): 알림 메시지
        """
        try:
            bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
            
            for chat_id in self.alert_recipients['telegram']:
                bot.send_message(chat_id=chat_id, text=message)
                
        except Exception as e:
            self.logger.error(f"텔레그램 알림 전송 실패: {str(e)}")
            
    def get_resource_usage_chart(self, hours: int = 24) -> go.Figure:
        """
        리소스 사용량 차트 생성
        
        Args:
            hours (int): 시간 범위
            
        Returns:
            go.Figure: 리소스 사용량 차트
        """
        try:
            metrics = self.db_manager.get_metrics(
                start_time=datetime.now() - timedelta(hours=hours)
            )
            
            if not metrics:
                return None
                
            fig = make_subplots(rows=2, cols=2, subplot_titles=('CPU 사용량', '메모리 사용량', '디스크 사용량', '네트워크 사용량'))
            
            # CPU 사용량
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['resource_usage']['cpu'] for m in metrics],
                    name='CPU'
                ),
                row=1, col=1
            )
            
            # 메모리 사용량
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['resource_usage']['memory'] for m in metrics],
                    name='Memory'
                ),
                row=1, col=2
            )
            
            # 디스크 사용량
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['resource_usage']['disk'] for m in metrics],
                    name='Disk'
                ),
                row=2, col=1
            )
            
            # 네트워크 사용량
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['resource_usage']['network'] for m in metrics],
                    name='Network'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="리소스 사용량 모니터링")
            return fig
            
        except Exception as e:
            self.logger.error(f"리소스 사용량 차트 생성 실패: {str(e)}")
            return None
            
    def get_api_performance_chart(self, hours: int = 24) -> go.Figure:
        """
        API 성능 차트 생성
        
        Args:
            hours (int): 시간 범위
            
        Returns:
            go.Figure: API 성능 차트
        """
        try:
            metrics = self.db_manager.get_metrics(
                start_time=datetime.now() - timedelta(hours=hours)
            )
            
            if not metrics:
                return None
                
            fig = go.Figure()
            
            # 응답 시간
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['api_performance']['response_time'] for m in metrics],
                    name='응답 시간'
                )
            )
            
            # 성공률
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['api_performance']['success'] for m in metrics],
                    name='성공률',
                    yaxis='y2'
                )
            )
            
            fig.update_layout(
                title="API 성능 모니터링",
                yaxis=dict(title="응답 시간 (ms)"),
                yaxis2=dict(title="성공률", overlaying="y", side="right")
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"API 성능 차트 생성 실패: {str(e)}")
            return None
            
    def get_processing_speed_chart(self, hours: int = 24) -> go.Figure:
        """
        처리 속도 차트 생성
        
        Args:
            hours (int): 시간 범위
            
        Returns:
            go.Figure: 처리 속도 차트
        """
        try:
            metrics = self.db_manager.get_metrics(
                start_time=datetime.now() - timedelta(hours=hours)
            )
            
            if not metrics:
                return None
                
            fig = go.Figure()
            
            # 처리 시간
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['processing_speed']['processing_time'] for m in metrics],
                    name='처리 시간'
                )
            )
            
            # 데이터 크기
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['processing_speed']['data_size'] for m in metrics],
                    name='데이터 크기',
                    yaxis='y2'
                )
            )
            
            fig.update_layout(
                title="데이터 처리 속도 모니터링",
                yaxis=dict(title="처리 시간 (ms)"),
                yaxis2=dict(title="데이터 크기", overlaying="y", side="right")
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"처리 속도 차트 생성 실패: {str(e)}")
            return None
            
    def get_system_status(self) -> Dict:
        """
        시스템 상태 조회
        
        Returns:
            Dict: 시스템 상태
        """
        try:
            # 최근 메트릭 조회
            metrics = self.db_manager.get_metrics(limit=1)
            
            if not metrics:
                return {}
                
            latest = metrics[0]
            
            # 상태 평가
            status = {
                'resource_usage': {
                    'cpu': {
                        'value': latest['resource_usage']['cpu'],
                        'status': 'normal' if latest['resource_usage']['cpu'] < self.alert_thresholds['cpu'] else 'warning'
                    },
                    'memory': {
                        'value': latest['resource_usage']['memory'],
                        'status': 'normal' if latest['resource_usage']['memory'] < self.alert_thresholds['memory'] else 'warning'
                    },
                    'disk': {
                        'value': latest['resource_usage']['disk'],
                        'status': 'normal' if latest['resource_usage']['disk'] < self.alert_thresholds['disk'] else 'warning'
                    }
                },
                'api_performance': {
                    'response_time': {
                        'value': latest['api_performance']['response_time'],
                        'status': 'normal' if latest['api_performance']['response_time'] < self.alert_thresholds['api_response'] else 'warning'
                    },
                    'success_rate': {
                        'value': latest['api_performance']['success'],
                        'status': 'normal' if latest['api_performance']['success'] else 'warning'
                    }
                },
                'processing_speed': {
                    'processing_time': {
                        'value': latest['processing_speed']['processing_time'],
                        'status': 'normal' if latest['processing_speed']['processing_time'] < self.alert_thresholds['processing_speed'] else 'warning'
                    }
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"시스템 상태 조회 실패: {str(e)}")
            return {}
            
    def set_alert_thresholds(self, thresholds: Dict):
        """
        알림 임계값 설정
        
        Args:
            thresholds (Dict): 임계값 설정
        """
        self.alert_thresholds.update(thresholds)
        
    def set_alert_channels(self, channels: Dict):
        """
        알림 채널 설정
        
        Args:
            channels (Dict): 채널 설정
        """
        self.alert_channels.update(channels)
        
    def set_alert_recipients(self, recipients: Dict):
        """
        알림 수신자 설정
        
        Args:
            recipients (Dict): 수신자 설정
        """
        self.alert_recipients.update(recipients) 