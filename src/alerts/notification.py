import logging
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

class NotificationType(Enum):
    """알림 유형"""
    TRADE_SIGNAL = "trade_signal"
    RISK_WARNING = "risk_warning"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_UPDATE = "performance_update"
    POSITION_UPDATE = "position_update"

class NotificationPriority(Enum):
    """알림 우선순위"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """알림 채널"""
    EMAIL = "email"
    SMS = "sms"
    TELEGRAM = "telegram"
    SLACK = "slack"
    WEBHOOK = "webhook"

class Notification:
    """알림 클래스"""
    
    def __init__(self,
                 title: str,
                 message: str,
                 priority: NotificationPriority = NotificationPriority.MEDIUM,
                 channels: List[NotificationChannel] = None,
                 metadata: Dict[str, Any] = None):
        """
        알림 초기화
        
        Args:
            title: 알림 제목
            message: 알림 메시지
            priority: 알림 우선순위
            channels: 알림 채널 목록
            metadata: 추가 메타데이터
        """
        self.title = title
        self.message = message
        self.priority = priority
        self.channels = channels or [NotificationChannel.EMAIL]
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.status = "pending"
        self.retry_count = 0
        self.max_retries = 3
        
class NotificationManager:
    """알림 관리 시스템"""
    
    def __init__(self,
                 config_path: str = "./config/notification_config.json",
                 log_dir: str = "./logs"):
        """
        알림 관리 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("notification_manager")
        
        # 설정 로드
        self.config = self._load_config()
        
        # 알림 큐
        self.notification_queue = queue.PriorityQueue()
        
        # 알림 처리 스레드
        self.processing_thread = None
        self.is_processing = False
        
        # 알림 통계
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "by_channel": {},
            "by_priority": {}
        }
        
        # 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def start(self) -> None:
        """알림 시스템 시작"""
        try:
            self.is_processing = True
            self.processing_thread = threading.Thread(
                target=self._process_notifications,
                daemon=True
            )
            self.processing_thread.start()
            
            self.logger.info("알림 시스템 시작")
            
        except Exception as e:
            self.logger.error(f"알림 시스템 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """알림 시스템 중지"""
        try:
            self.is_processing = False
            if self.processing_thread:
                self.processing_thread.join()
                
            self.logger.info("알림 시스템 중지")
            
        except Exception as e:
            self.logger.error(f"알림 시스템 중지 중 오류 발생: {e}")
            raise
            
    def send_notification(self, notification: Notification) -> None:
        """
        알림 전송 요청
        
        Args:
            notification: 전송할 알림
        """
        try:
            # 우선순위에 따른 큐 우선순위 설정
            priority_map = {
                NotificationPriority.LOW: 3,
                NotificationPriority.MEDIUM: 2,
                NotificationPriority.HIGH: 1,
                NotificationPriority.CRITICAL: 0
            }
            
            # 알림 큐에 추가
            self.notification_queue.put((
                priority_map[notification.priority],
                time.time(),
                notification
            ))
            
            self.logger.info(f"알림 전송 요청: {notification.title}")
            
        except Exception as e:
            self.logger.error(f"알림 전송 요청 중 오류 발생: {e}")
            
    def _process_notifications(self) -> None:
        """알림 처리"""
        try:
            while self.is_processing:
                if not self.notification_queue.empty():
                    # 알림 큐에서 알림 가져오기
                    _, _, notification = self.notification_queue.get()
                    
                    # 알림 전송
                    success = self._send_notification_to_channels(notification)
                    
                    if success:
                        notification.status = "sent"
                        self.stats["total_sent"] += 1
                    else:
                        notification.status = "failed"
                        self.stats["total_failed"] += 1
                        
                    # 통계 업데이트
                    self._update_stats(notification)
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"알림 처리 중 오류 발생: {e}")
            
    def _send_notification_to_channels(self,
                                     notification: Notification) -> bool:
        """
        채널별 알림 전송
        
        Args:
            notification: 전송할 알림
            
        Returns:
            전송 성공 여부
        """
        try:
            success = True
            
            for channel in notification.channels:
                if channel == NotificationChannel.EMAIL:
                    if not self._send_email(notification):
                        success = False
                elif channel == NotificationChannel.SMS:
                    if not self._send_sms(notification):
                        success = False
                elif channel == NotificationChannel.TELEGRAM:
                    if not self._send_telegram(notification):
                        success = False
                elif channel == NotificationChannel.SLACK:
                    if not self._send_slack(notification):
                        success = False
                elif channel == NotificationChannel.WEBHOOK:
                    if not self._send_webhook(notification):
                        success = False
                        
            return success
            
        except Exception as e:
            self.logger.error(f"채널별 알림 전송 중 오류 발생: {e}")
            return False
            
    def _send_email(self, notification: Notification) -> bool:
        """
        이메일 전송
        
        Args:
            notification: 전송할 알림
            
        Returns:
            전송 성공 여부
        """
        try:
            # 이메일 설정
            smtp_config = self.config.get("email", {})
            smtp_server = smtp_config.get("server")
            smtp_port = smtp_config.get("port")
            smtp_username = smtp_config.get("username")
            smtp_password = smtp_config.get("password")
            
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg["From"] = smtp_username
            msg["To"] = notification.metadata.get("recipient")
            msg["Subject"] = notification.title
            
            # 메시지 본문 추가
            msg.attach(MIMEText(notification.message, "plain"))
            
            # SMTP 서버 연결 및 이메일 전송
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            self.logger.error(f"이메일 전송 중 오류 발생: {e}")
            return False
            
    def _send_sms(self, notification: Notification) -> bool:
        """
        SMS 전송
        
        Args:
            notification: 전송할 알림
            
        Returns:
            전송 성공 여부
        """
        try:
            # SMS 설정
            sms_config = self.config.get("sms", {})
            api_key = sms_config.get("api_key")
            api_secret = sms_config.get("api_secret")
            sender = sms_config.get("sender")
            
            # SMS API 호출
            url = "https://api.sms.com/send"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "to": notification.metadata.get("phone"),
                "from": sender,
                "text": f"{notification.title}\n{notification.message}"
            }
            
            response = requests.post(url, headers=headers, json=data)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"SMS 전송 중 오류 발생: {e}")
            return False
            
    def _send_telegram(self, notification: Notification) -> bool:
        """
        텔레그램 메시지 전송
        
        Args:
            notification: 전송할 알림
            
        Returns:
            전송 성공 여부
        """
        try:
            # 텔레그램 설정
            telegram_config = self.config.get("telegram", {})
            bot_token = telegram_config.get("bot_token")
            chat_id = notification.metadata.get("chat_id")
            
            # 텔레그램 API 호출
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": f"*{notification.title}*\n{notification.message}",
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=data)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"텔레그램 메시지 전송 중 오류 발생: {e}")
            return False
            
    def _send_slack(self, notification: Notification) -> bool:
        """
        슬랙 메시지 전송
        
        Args:
            notification: 전송할 알림
            
        Returns:
            전송 성공 여부
        """
        try:
            # 슬랙 설정
            slack_config = self.config.get("slack", {})
            webhook_url = slack_config.get("webhook_url")
            
            # 슬랙 메시지 생성
            message = {
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": notification.title
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": notification.message
                        }
                    }
                ]
            }
            
            # 슬랙 API 호출
            response = requests.post(webhook_url, json=message)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"슬랙 메시지 전송 중 오류 발생: {e}")
            return False
            
    def _send_webhook(self, notification: Notification) -> bool:
        """
        웹훅 전송
        
        Args:
            notification: 전송할 알림
            
        Returns:
            전송 성공 여부
        """
        try:
            # 웹훅 설정
            webhook_url = notification.metadata.get("webhook_url")
            
            # 웹훅 데이터 생성
            data = {
                "title": notification.title,
                "message": notification.message,
                "priority": notification.priority.value,
                "timestamp": notification.timestamp.isoformat(),
                "metadata": notification.metadata
            }
            
            # 웹훅 API 호출
            response = requests.post(webhook_url, json=data)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"웹훅 전송 중 오류 발생: {e}")
            return False
            
    def _update_stats(self, notification: Notification) -> None:
        """
        알림 통계 업데이트
        
        Args:
            notification: 처리된 알림
        """
        try:
            # 채널별 통계 업데이트
            for channel in notification.channels:
                if channel.value not in self.stats["by_channel"]:
                    self.stats["by_channel"][channel.value] = 0
                self.stats["by_channel"][channel.value] += 1
                
            # 우선순위별 통계 업데이트
            priority = notification.priority.value
            if priority not in self.stats["by_priority"]:
                self.stats["by_priority"][priority] = 0
            self.stats["by_priority"][priority] += 1
            
        except Exception as e:
            self.logger.error(f"알림 통계 업데이트 중 오류 발생: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """
        알림 통계 조회
        
        Returns:
            알림 통계
        """
        return self.stats 