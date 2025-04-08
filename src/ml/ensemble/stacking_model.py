from sklearn.base import BaseEstimator
import numpy as np
from typing import List, Dict
import lightgbm as lgb
import xgboost as xgb

class StackingEnsemble(BaseEstimator):
    def __init__(self, base_models: List, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or lgb.LGBMRegressor()
        self.base_predictions = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """베이스 모델과 메타 모델 학습"""
        self.base_predictions = np.column_stack([
            model.fit(X, y).predict(X)
            for model in self.base_models
        ])
        self.meta_model.fit(self.base_predictions, y)
        return self
