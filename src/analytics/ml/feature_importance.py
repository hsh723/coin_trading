from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass

@dataclass
class FeatureImportance:
    features: List[str]
    importance_scores: np.ndarray
    ranked_features: List[tuple]

class FeatureImportanceAnalyzer:
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)
        
    def analyze_importance(self, features: pd.DataFrame, target: pd.Series) -> FeatureImportance:
        """특성 중요도 분석"""
        self.model.fit(features, target)
        importance_scores = self.model.feature_importances_
        
        ranked_features = sorted(
            zip(features.columns, importance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return FeatureImportance(
            features=features.columns.tolist(),
            importance_scores=importance_scores,
            ranked_features=ranked_features
        )
