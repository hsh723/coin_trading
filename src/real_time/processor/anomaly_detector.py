import asyncio
from typing import Dict, List
import numpy as np

class AnomalyDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'detection_window': 100,
            'threshold': 3.0,  # 표준편차 기준
            'min_samples': 30
        }
        
    async def detect_anomalies(self, market_data: Dict) -> Dict:
        """실시간 이상 징후 감지"""
        price_anomalies = self._detect_price_anomalies(market_data['price'])
        volume_anomalies = self._detect_volume_anomalies(market_data['volume'])
        pattern_anomalies = self._detect_pattern_anomalies(market_data)
        
        return {
            'price_anomalies': price_anomalies,
            'volume_anomalies': volume_anomalies,
            'pattern_anomalies': pattern_anomalies,
            'anomaly_score': self._calculate_anomaly_score(price_anomalies, volume_anomalies)
        }
