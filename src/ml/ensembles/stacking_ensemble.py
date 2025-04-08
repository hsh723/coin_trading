from sklearn.base import BaseEstimator
import numpy as np
from typing import List, Dict
import lightgbm as lgb
import xgboost as xgb

class StackingEnsemble:
    def __init__(self, base_models: List[BaseEstimator], 
                 meta_model: BaseEstimator = None):
        self.base_models = base_models
        self.meta_model = meta_model or lgb.LGBMRegressor()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """앙상블 모델 학습"""
        base_predictions = np.column_stack([
            model.fit(X, y).predict(X)
            for model in self.base_models
        ])
        self.meta_model.fit(base_predictions, y)
