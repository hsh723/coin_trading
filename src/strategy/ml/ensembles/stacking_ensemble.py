from typing import List, Dict
import numpy as np
from sklearn.base import BaseEstimator

class StackingEnsemble:
    def __init__(self, base_models: List[BaseEstimator], meta_model: BaseEstimator):
        self.base_models = base_models
        self.meta_model = meta_model
        
    async def fit_ensemble(self, X: np.ndarray, y: np.ndarray):
        """앙상블 학습"""
        base_predictions = np.column_stack([
            model.fit(X, y).predict(X) for model in self.base_models
        ])
        self.meta_model.fit(base_predictions, y)
