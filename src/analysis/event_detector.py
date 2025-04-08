import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class MarketEvent:
    type: str
    timestamp: pd.Timestamp
    severity: float
    details: Dict

class EventDetector:
    def __init__(self, config: Dict):
        self.thresholds = config['thresholds']
        self.window_size = config['window_size']
        self.events = []
        
    def detect_anomalies(self, data: pd.DataFrame) -> List[MarketEvent]:
        """이상 현상 감지"""
        volatility_events = self._detect_volatility_anomalies(data)
        volume_events = self._detect_volume_anomalies(data)
        correlation_events = self._detect_correlation_breaks(data)
        
        return volatility_events + volume_events + correlation_events
