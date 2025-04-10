from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class AnomalyDetection:
    anomaly_type: str
    confidence: float
    affected_metrics: List[str]
    threshold_breaches: Dict[str, float]

class StrategyAnomalyDetector:
    def __init__(self, detection_config: Dict = None):
        self.config = detection_config or {
            'z_score_threshold': 3.0,
            'window_size': 100,
            'sensitivity': 0.8
        }
        
    async def detect_anomalies(self, 
                             strategy_metrics: Dict,
                             historical_data: pd.DataFrame) -> List[AnomalyDetection]:
        """전략 이상 징후 감지"""
        anomalies = []
        
        # 주요 메트릭스에 대한 이상 감지
        for metric, value in strategy_metrics.items():
            if self._is_anomalous(value, historical_data[metric]):
                anomalies.append(self._create_anomaly_detection(metric, value))
                
        return anomalies
