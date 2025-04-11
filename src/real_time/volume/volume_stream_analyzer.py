import asyncio
from typing import Dict, List
import numpy as np

class VolumeStreamAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'volume_window': 100,
            'update_interval': 0.1,
            'alert_threshold': 2.0
        }
        self.volume_buffer = []
        
    async def analyze_volume_stream(self, volume_data: Dict) -> Dict:
        """실시간 거래량 스트림 분석"""
        self.volume_buffer.append(volume_data['volume'])
        if len(self.volume_buffer) > self.config['volume_window']:
            self.volume_buffer.pop(0)
            
        return {
            'current_volume': volume_data['volume'],
            'volume_ma': np.mean(self.volume_buffer),
            'volume_std': np.std(self.volume_buffer),
            'volume_alerts': self._check_volume_alerts(),
            'trend_analysis': self._analyze_volume_trend()
        }
