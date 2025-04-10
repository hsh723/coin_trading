from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Channel:
    channel_id: str
    type: str  # email, slack, telegram
    config: Dict
    enabled: bool

class AlertChannelManager:
    def __init__(self):
        self.channels = {}
        self.default_channels = ['email']
        
    async def send_to_channels(self, alert: Dict, channels: List[str] = None) -> Dict[str, bool]:
        """여러 채널로 알림 전송"""
        channels = channels or self.default_channels
        results = {}
        
        for channel_id in channels:
            if channel_id in self.channels and self.channels[channel_id].enabled:
                success = await self._send_to_channel(
                    self.channels[channel_id], 
                    alert
                )
                results[channel_id] = success
                
        return results
