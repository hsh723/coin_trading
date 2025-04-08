import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from sklearn.ensemble import GradientBoostingRegressor

@dataclass
class PortfolioPrediction:
    asset_returns: Dict[str, float]
    confidence_scores: Dict[str, float]
    risk_metrics: Dict[str, float]
    allocation_advice: Dict[str, float]

class PortfolioPredictor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'prediction_horizon': 5,
            'confidence_threshold': 0.7,
            'risk_threshold': 0.02
        }
        self.models = {}
        
    def predict_portfolio_returns(self, 
                                features: Dict[str, pd.DataFrame]) -> PortfolioPrediction:
        """포트폴리오 수익률 예측"""
        predictions = {}
        confidence = {}
        
        for asset, asset_features in features.items():
            if asset not in self.models:
                self.models[asset] = self._train_asset_model(asset_features)
            
            pred = self.models[asset].predict(asset_features)
            predictions[asset] = pred[-1]
            confidence[asset] = self._calculate_confidence(pred)
            
        return PortfolioPrediction(
            asset_returns=predictions,
            confidence_scores=confidence,
            risk_metrics=self._calculate_risk_metrics(predictions),
            allocation_advice=self._generate_allocation_advice(predictions, confidence)
        )
