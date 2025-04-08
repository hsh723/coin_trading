import numpy as np
from typing import List, Dict
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass

@dataclass
class AnomalyResult:
    timestamp: pd.Timestamp
    score: float
    is_anomaly: bool
    features: Dict[str, float]

class AnomalyDetector:
    def __init__(self, config: Dict):
        self.isolation_forest = IsolationForest(
            contamination=config.get('contamination', 0.1),
            random_state=42
        )
        self.threshold = config.get('threshold', -0.5)
        
    def detect_anomalies(self, market_data: pd.DataFrame) -> List[AnomalyResult]:
        """이상치 감지"""
        features = self._extract_features(market_data)
        scores = self.isolation_forest.fit_predict(features)
        
        return [
            AnomalyResult(
                timestamp=market_data.index[i],
                score=float(scores[i]),
                is_anomaly=scores[i] < self.threshold,
                features=dict(zip(features.columns, features.iloc[i]))
            )
            for i in range(len(scores))
        ]
