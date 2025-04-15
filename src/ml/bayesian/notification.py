import logging
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class NotificationType:
    """알림 유형"""
    TRADE_SIGNAL = "trade_signal"
    RISK_WARNING = "risk_warning"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_UPDATE = "performance_update"
    POSITION_UPDATE = "position_update"

class NotificationPriority:
    """알림 우선순위"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel:
    """알림 채널"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    SLACK = "slack"

class NotificationManager:
    """알림 관리자"""
    def __init__(self,
                 config_path: str = "./config/notification_config.json",
                 log_dir: str = "./logs"):
        """
        알림 관리자 초기화
        
        Args:
            config_path: 설정 파일 경로
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.log_dir = log_dir
        
        # 설정 로드
        self.config = self._load_config()
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        # 알림 큐
        self.notification_queue = asyncio.Queue()
        
        # 알림 처리 태스크
        self.processing_task = None
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("notification_manager")
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        log_file = os.path.join(self.log_dir, "notifications.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
        
    async def start(self):
        """알림 시스템 시작"""
        self.processing_task = asyncio.create_task(self._process_notifications())
        self.logger.info("알림 시스템 시작")
        
    async def stop(self):
        """알림 시스템 중지"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("알림 시스템 중지")
        
    async def send_notification(self,
                              notification_type: str,
                              message: str,
                              priority: str = NotificationPriority.MEDIUM,
                              channels: Optional[List[str]] = None):
        """
        알림 전송
        
        Args:
            notification_type: 알림 유형
            message: 알림 메시지
            priority: 우선순위
            channels: 알림 채널 목록
        """
        notification = {
            'type': notification_type,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now().isoformat(),
            'channels': channels or self.config.get('default_channels', [])
        }
        
        await self.notification_queue.put(notification)
        self.logger.info(f"알림 큐에 추가: {notification}")
        
    async def _process_notifications(self):
        """알림 처리"""
        while True:
            try:
                notification = await self.notification_queue.get()
                
                # 알림 전송
                for channel in notification['channels']:
                    try:
                        if channel == NotificationChannel.TELEGRAM:
                            await self._send_telegram(notification)
                        elif channel == NotificationChannel.EMAIL:
                            await self._send_email(notification)
                        elif channel == NotificationChannel.SLACK:
                            await self._send_slack(notification)
                    except Exception as e:
                        self.logger.error(f"채널 {channel} 알림 전송 중 오류 발생: {e}")
                        
                # 알림 기록
                self._log_notification(notification)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"알림 처리 중 오류 발생: {e}")
                
    async def _send_telegram(self, notification: Dict[str, Any]):
        """텔레그램 알림 전송"""
        bot_token = self.config.get('telegram', {}).get('bot_token')
        chat_id = self.config.get('telegram', {}).get('chat_id')
        
        if not bot_token or not chat_id:
            raise ValueError("텔레그램 설정이 없습니다")
            
        message = self._format_message(notification)
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }) as response:
                if response.status != 200:
                    raise Exception(f"텔레그램 API 오류: {await response.text()}")
                    
    async def _send_email(self, notification: Dict[str, Any]):
        """이메일 알림 전송"""
        smtp_config = self.config.get('email', {})
        
        if not all(k in smtp_config for k in ['smtp_server', 'smtp_port', 'username', 'password', 'from_email', 'to_email']):
            raise ValueError("이메일 설정이 없습니다")
            
        message = MIMEMultipart()
        message['From'] = smtp_config['from_email']
        message['To'] = smtp_config['to_email']
        message['Subject'] = f"[{notification['priority'].upper()}] {notification['type']}"
        
        body = self._format_message(notification)
        message.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(message)
            
    async def _send_slack(self, notification: Dict[str, Any]):
        """슬랙 알림 전송"""
        slack_config = self.config.get('slack', {})
        
        if not slack_config.get('bot_token') or not slack_config.get('channel'):
            raise ValueError("슬랙 설정이 없습니다")
            
        client = WebClient(token=slack_config['bot_token'])
        message = self._format_message(notification)
        
        try:
            response = client.chat_postMessage(
                channel=slack_config['channel'],
                text=message
            )
            if not response['ok']:
                raise Exception(f"슬랙 API 오류: {response['error']}")
        except SlackApiError as e:
            raise Exception(f"슬랙 API 오류: {e.response['error']}")
            
    def _format_message(self, notification: Dict[str, Any]) -> str:
        """알림 메시지 포맷팅"""
        priority_emoji = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.MEDIUM: "⚠️",
            NotificationPriority.HIGH: "🚨",
            NotificationPriority.CRITICAL: "🔥"
        }
        
        return f"""
{priority_emoji.get(notification['priority'], '')} <b>{notification['type'].upper()}</b>
우선순위: {notification['priority'].upper()}
시간: {notification['timestamp']}

{notification['message']}
"""
        
    def _log_notification(self, notification: Dict[str, Any]):
        """알림 기록"""
        log_file = os.path.join(self.log_dir, f"notifications_{datetime.now().strftime('%Y%m%d')}.log")
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(notification) + '\n')
            
    def get_notification_history(self,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               notification_type: Optional[str] = None,
                               priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        알림 기록 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            notification_type: 알림 유형
            priority: 우선순위
            
        Returns:
            알림 기록 목록
        """
        notifications = []
        
        # 로그 파일 목록
        log_files = [f for f in os.listdir(self.log_dir) if f.startswith('notifications_')]
        
        for log_file in log_files:
            with open(os.path.join(self.log_dir, log_file), 'r') as f:
                for line in f:
                    notification = json.loads(line)
                    
                    # 필터링
                    if start_time and datetime.fromisoformat(notification['timestamp']) < start_time:
                        continue
                    if end_time and datetime.fromisoformat(notification['timestamp']) > end_time:
                        continue
                    if notification_type and notification['type'] != notification_type:
                        continue
                    if priority and notification['priority'] != priority:
                        continue
                        
                    notifications.append(notification)
                    
        return sorted(notifications, key=lambda x: x['timestamp'], reverse=True) 