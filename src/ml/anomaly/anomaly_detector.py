import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List

class AnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.score_threshold = -0.5
        
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, List]:
        """이상치 감지"""
        scores = self.model.fit_predict(data)
        anomaly_indices = np.where(scores == -1)[0]
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': self.model.score_samples(data).tolist(),
            'features_contribution': self._calculate_feature_contribution(data, anomaly_indices)
        }
