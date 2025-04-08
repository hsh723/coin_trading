import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from typing import List

class FeatureSelector:
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
        self.selector = SelectKBest(score_func=f_regression, k=n_features)
        self.selected_features = []
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """중요 특성 선택"""
        self.selector.fit(X, y)
        feature_mask = self.selector.get_support()
        self.selected_features = X.columns[feature_mask].tolist()
        return X[self.selected_features]
