import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeAnomaly:
    is_anomaly: bool
    anomaly_score: float
    detection_time: str
    anomaly_type: str

class VolumeAnomalyDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'z_score_threshold': 3.0,
            'window_size': 50,
            'min_samples': 30
        }
        
    async def detect_anomalies(self, volume_data: np.ndarray) -> List[VolumeAnomaly]:
        """거래량 이상 징후 감지"""
        z_scores = self._calculate_z_scores(volume_data)
        anomalies = self._identify_anomalies(z_scores)
        
        return [
            VolumeAnomaly(
                is_anomaly=True,
                anomaly_score=score,
                detection_time=time,
                anomaly_type=self._classify_anomaly(score)
            )
            for time, score in anomalies
        ]
