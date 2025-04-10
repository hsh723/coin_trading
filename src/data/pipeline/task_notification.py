from typing import Dict, List
from dataclasses import dataclass

@dataclass
class NotificationConfig:
    channels: List[str]
    severity_levels: Dict[str, int]
    templates: Dict[str, str]

class TaskNotificationSystem:
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.notification_history = []
        
    async def send_notification(self, 
                              message: str,
                              severity: str = 'info',
                              metadata: Dict = None) -> bool:
        """작업 알림 전송"""
        if severity not in self.config.severity_levels:
            return False
            
        notification = {
            'message': message,
            'severity': severity,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.notification_history.append(notification)
        
        for channel in self.config.channels:
            await self._send_to_channel(channel, notification)
            
        return True
