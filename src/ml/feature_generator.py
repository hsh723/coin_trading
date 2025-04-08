import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class FeatureSet:
    technical_features: pd.DataFrame
    market_features: pd.DataFrame
    temporal_features: pd.DataFrame
    combined_features: pd.DataFrame

class FeatureGenerator:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'window_sizes': [5, 10, 20, 50],
            'use_ta_features': True,
            'use_market_features': True,
            'use_temporal_features': True
        }
        
    def generate_features(self, market_data: pd.DataFrame) -> FeatureSet:
        """특성 생성"""
        technical_features = self._generate_technical_features(market_data)
        market_features = self._generate_market_features(market_data)
        temporal_features = self._generate_temporal_features(market_data)
        
        return FeatureSet(
            technical_features=technical_features,
            market_features=market_features,
            temporal_features=temporal_features,
            combined_features=pd.concat([
                technical_features,
                market_features,
                temporal_features
            ], axis=1)
        )
