import asyncio
from typing import Dict, List
import numpy as np

class RealtimeVolumeTracker:
    def __init__(self, tracking_window: int = 100):
        self.tracking_window = tracking_window
        self.volume_buffer = []
        
    async def track_volume(self, market_data: Dict) -> Dict:
        """실시간 거래량 추적"""
        self.volume_buffer.append(market_data['volume'])
        if len(self.volume_buffer) > self.tracking_window:
            self.volume_buffer.pop(0)
            
        return {
            'current_volume': market_data['volume'],
            'volume_ma': np.mean(self.volume_buffer),
            'volume_trend': self._detect_trend(),
            'abnormal_signals': self._detect_abnormal_volume()
        }
