from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class Acknowledgment:
    alert_id: str
    user_id: str
    timestamp: float
    comment: str
    status: str

class AlertAcknowledgmentManager:
    def __init__(self):
        self.acknowledgments = {}
        self.pending_alerts = set()
        
    async def acknowledge_alert(self, alert_id: str, 
                              user_id: str, 
                              comment: str = "") -> Acknowledgment:
        """알림 승인 처리"""
        if alert_id not in self.pending_alerts:
            raise ValueError(f"Alert {alert_id} not found or already acknowledged")
            
        ack = Acknowledgment(
            alert_id=alert_id,
            user_id=user_id,
            timestamp=time.time(),
            comment=comment,
            status='acknowledged'
        )
        
        self.acknowledgments[alert_id] = ack
        self.pending_alerts.remove(alert_id)
        
        return ack
