import numpy as np
import pandas as pd
from typing import Dict, List
from .base import BaseStrategy
from ..analysis.machine_learning import MLAnalyzer
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier

@dataclass
class MLPrediction:
    signal: str
    probability: float
    features_importance: Dict[str, float]
    model_confidence: float

class MachineLearningStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.ml_analyzer = MLAnalyzer()
        self.lookback_period = config.get('lookback_period', 60)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100)
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """특성 엔지니어링"""
        features = []
        for column in ['open', 'high', 'low', 'close', 'volume']:
            if column in data.columns:
                features.append(data[column].pct_change())
                features.append(data[column].rolling(window=self.lookback_period).mean())
                features.append(data[column].rolling(window=self.lookback_period).std())
        
        return np.column_stack(features)

    async def generate_prediction(self, market_data: pd.DataFrame) -> MLPrediction:
        """ML 기반 예측 생성"""
        features = self._extract_features(market_data)
        prediction = self.model.predict_proba(features)
        
        return MLPrediction(
            signal=self._convert_prediction_to_signal(prediction),
            probability=np.max(prediction),
            features_importance=dict(zip(
                self.config['features'],
                self.model.feature_importances_
            )),
            model_confidence=self._calculate_model_confidence()
        )
