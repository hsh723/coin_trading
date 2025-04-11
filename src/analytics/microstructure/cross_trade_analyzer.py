import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CrossTradeMetrics:
    cross_trade_ratio: float
    wash_trade_probability: float
    suspicious_patterns: List[Dict]
    trade_clusters: Dict[str, float]

class CrossTradeAnalyzer:
    def __init__(self, detection_config: Dict = None):
        self.config = detection_config or {
            'time_threshold': 0.1,  # seconds
            'price_threshold': 0.0001
        }
        
    async def analyze_cross_trades(self, trades: List[Dict]) -> CrossTradeMetrics:
        """교차 거래 분석"""
        suspicious = self._detect_suspicious_trades(trades)
        clusters = self._analyze_trade_clusters(trades)
        
        return CrossTradeMetrics(
            cross_trade_ratio=self._calculate_cross_ratio(suspicious, trades),
            wash_trade_probability=self._estimate_wash_probability(suspicious),
            suspicious_patterns=suspicious,
            trade_clusters=clusters
        )
