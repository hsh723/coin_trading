import numpy as np
import pandas as pd
from typing import Dict, List
from .base import BaseStrategy
from ..analysis.machine_learning import MLAnalyzer
from sklearn.preprocessing import StandardScaler

class MachineLearningStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.ml_analyzer = MLAnalyzer()
        self.lookback_period = config.get('lookback_period', 60)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """특성 엔지니어링"""
        features = []
        for column in ['open', 'high', 'low', 'close', 'volume']:
            if column in data.columns:
                features.append(data[column].pct_change())
                features.append(data[column].rolling(window=self.lookback_period).mean())
                features.append(data[column].rolling(window=self.lookback_period).std())
        
        return np.column_stack(features)
