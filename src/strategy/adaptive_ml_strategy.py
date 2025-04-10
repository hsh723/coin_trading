from typing import Dict, List
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

@dataclass
class MLPredictionResult:
    predicted_price: float
    confidence: float
    feature_importance: Dict[str, float]
    model_metrics: Dict[str, float]

class AdaptiveMLStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'prediction_horizon': 24,
            'retrain_interval': 168,  # 7일
            'min_samples': 1000
        }
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1
        )
        
    async def predict_next_move(self, market_data: pd.DataFrame) -> MLPredictionResult:
        """다음 가격 움직임 예측"""
        features = self._extract_features(market_data)
        prediction = self.model.predict(features)
        
        return MLPredictionResult(
            predicted_price=prediction[0],
            confidence=self._calculate_confidence(prediction),
            feature_importance=dict(zip(
                self.feature_names_,
                self.model.feature_importances_
            )),
            model_metrics=self._evaluate_model()
        )
