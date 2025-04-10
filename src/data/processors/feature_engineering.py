import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class FeatureSet:
    technical_features: pd.DataFrame
    market_features: pd.DataFrame
    time_features: pd.DataFrame

class FeatureEngineer:
    def __init__(self, feature_config: Dict = None):
        self.feature_config = feature_config or {
            'technical': True,
            'market': True,
            'time': True
        }
        
    async def generate_features(self, data: pd.DataFrame) -> FeatureSet:
        """특성 생성"""
        features = {}
        
        if self.feature_config['technical']:
            features['technical'] = self._create_technical_features(data)
            
        if self.feature_config['market']:
            features['market'] = self._create_market_features(data)
            
        if self.feature_config['time']:
            features['time'] = self._create_time_features(data)
            
        return FeatureSet(**features)
