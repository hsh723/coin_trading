from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Any
import lightgbm as lgb
import numpy as np

class ModelTrainer:
    def __init__(self, model_config: Dict = None):
        self.model_config = model_config or {
            'model_type': 'lightgbm',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05
        }
        
    def train_model(self, features: pd.DataFrame, 
                   targets: pd.Series) -> Dict[str, Any]:
        """모델 학습"""
        tscv = TimeSeriesSplit(n_splits=5)
        models = []
        
        for train_idx, val_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            y_train = targets.iloc[train_idx]
            X_val = features.iloc[val_idx]
            y_val = targets.iloc[val_idx]
            
            model = self._train_single_model(X_train, y_train, X_val, y_val)
            models.append(model)
            
        return {
            'models': models,
            'feature_importance': self._calculate_feature_importance(models)
        }
