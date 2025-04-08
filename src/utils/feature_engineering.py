import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    def __init__(self, feature_config: Dict = None):
        self.config = feature_config or {}
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 기반 특성 생성"""
        features = data.copy()
        
        # Price Features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Volume Features
        features['volume_ma'] = data['volume'].rolling(window=20).mean()
        features['volume_std'] = data['volume'].rolling(window=20).std()
        
        return features
