import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketPrediction:
    direction: str
    probability: float
    predicted_price: float
    confidence_interval: tuple

class MarketPredictor:
    def __init__(self, model_ensemble: Dict = None):
        self.models = model_ensemble or {}
        self.prediction_horizon = 24  # 24시간
        
    async def predict_market(self, features: pd.DataFrame) -> MarketPrediction:
        """시장 방향성 예측"""
        predictions = []
        for model_name, model in self.models.items():
            pred = await self._get_model_prediction(model, features)
            predictions.append(pred)
            
        ensemble_pred = np.mean(predictions, axis=0)
        return self._create_prediction_result(ensemble_pred)
