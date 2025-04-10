from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class NotificationConfig:
    channel: str
    template: str
    recipients: List[str]
    priority: str

class NotificationSender:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'use_tls': True
            },
            'telegram': {
                'bot_token': '',
                'chat_ids': []
            }
        }
        self.notification_queue = asyncio.Queue()
        
    async def send_notification(self, 
                              message: str, 
                              config: NotificationConfig) -> bool:
        """알림 전송"""
        try:
            if config.channel == 'email':
                return await self._send_email(message, config)
            elif config.channel == 'telegram':
                return await self._send_telegram(message, config)
            elif config.channel == 'slack':
                return await self._send_slack(message, config)
            return False
        except Exception as e:
            await self._handle_send_error(e, config)
            return False
