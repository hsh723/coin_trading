import numpy as np
from typing import List, Dict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb

class EnsemblePredictor:
    def __init__(self, config: Dict):
        self.models = {
            'rf': RandomForestRegressor(**config.get('rf_params', {})),
            'gbm': GradientBoostingRegressor(**config.get('gbm_params', {})),
            'lgb': lgb.LGBMRegressor(**config.get('lgb_params', {}))
        }
        self.weights = config.get('model_weights', {
            'rf': 0.3, 'gbm': 0.3, 'lgb': 0.4
        })
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """앙상블 예측 수행"""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(features)
            
        return self._combine_predictions(predictions)
