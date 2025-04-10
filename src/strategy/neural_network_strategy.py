from typing import Dict, List
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class NeuralPrediction:
    price_prediction: float
    confidence_score: float
    forecast_horizon: int
    prediction_intervals: Dict[str, float]

class NeuralNetworkStrategy:
    def __init__(self, model_config: Dict = None):
        self.config = model_config or {
            'input_size': 10,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        }
        self.model = self._build_model()
        
    async def predict_market(self, market_data: torch.Tensor) -> NeuralPrediction:
        """신경망 기반 시장 예측"""
        with torch.no_grad():
            prediction = self.model(market_data)
            confidence = self._calculate_prediction_confidence(prediction)
            
        return NeuralPrediction(
            price_prediction=prediction.item(),
            confidence_score=confidence,
            forecast_horizon=self.config.get('forecast_horizon', 24),
            prediction_intervals=self._calculate_prediction_intervals(prediction)
        )
