import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
import optuna

@dataclass
class TrainingResult:
    model: object
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    validation_predictions: pd.Series

class ModelTrainingPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'n_splits': 5,
            'test_size': 0.2,
            'optimization_trials': 100
        }
        
    def run_pipeline(self, features: pd.DataFrame, target: pd.Series) -> TrainingResult:
        """전체 훈련 파이프라인 실행"""
        # 데이터 전처리
        processed_features = self._preprocess_features(features)
        
        # 하이퍼파라미터 최적화
        best_params = self._optimize_hyperparameters(processed_features, target)
        
        # 최종 모델 훈련
        final_model = self._train_final_model(processed_features, target, best_params)
        
        return TrainingResult(
            model=final_model,
            metrics=self._calculate_metrics(final_model, processed_features, target),
            feature_importance=self._get_feature_importance(final_model, features.columns),
            validation_predictions=self._get_validation_predictions(final_model, processed_features)
        )
