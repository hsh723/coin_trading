import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class TrainingResult:
    model: object
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    validation_predictions: pd.Series

class ModelTrainingPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'test_size': 0.2,
            'n_splits': 5,
            'random_state': 42
        }
        self.scaler = StandardScaler()
        
    def run_pipeline(self, features: pd.DataFrame, target: pd.Series) -> TrainingResult:
        """ML 모델 훈련 파이프라인 실행"""
        # 데이터 전처리
        X_scaled = self._preprocess_features(features)
        
        # 시계열 교차 검증
        cv = TimeSeriesSplit(n_splits=self.config['n_splits'])
        models = []
        
        for train_idx, val_idx in cv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
            
            model = self._train_model(X_train, y_train, X_val, y_val)
            models.append(model)
            
        return self._create_training_result(models[-1], features, target)
