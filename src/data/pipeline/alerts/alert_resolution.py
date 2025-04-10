from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class Resolution:
    resolution_id: str
    alert_id: str
    resolver: str
    resolution_type: str
    comment: str
    timestamp: float

class AlertResolutionManager:
    def __init__(self):
        self.resolutions = {}
        self.resolution_types = ['fixed', 'invalid', 'deferred', 'duplicate']
        
    async def resolve_alert(self, alert_id: str, 
                          resolver: str, 
                          resolution_type: str,
                          comment: str = "") -> Resolution:
        """알림 해결 처리"""
        if resolution_type not in self.resolution_types:
            raise ValueError(f"Invalid resolution type: {resolution_type}")
            
        resolution = Resolution(
            resolution_id=self._generate_resolution_id(),
            alert_id=alert_id,
            resolver=resolver,
            resolution_type=resolution_type,
            comment=comment,
            timestamp=time.time()
        )
        
        self.resolutions[resolution.resolution_id] = resolution
        await self._update_alert_status(alert_id, resolution)
        
        return resolution
