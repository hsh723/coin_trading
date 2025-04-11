from typing import Dict, Any
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

class ModelTrainer:
    def __init__(self, model_config: Dict = None):
        self.config = model_config or {
            'model_type': 'lightgbm',
            'cv_folds': 5,
            'early_stopping': 50
        }
        
    async def train_model(self, 
                         features: np.ndarray, 
                         targets: np.ndarray) -> Any:
        """모델 학습"""
        if self.config['model_type'] == 'lightgbm':
            return self._train_lightgbm(features, targets)
        else:
            return self._train_gradient_boosting(features, targets)
