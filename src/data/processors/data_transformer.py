import pandas as pd
import numpy as np
from typing import Dict, List

class DataTransformer:
    def __init__(self, transformation_rules: Dict = None):
        self.transformation_rules = transformation_rules or {
            'price_scale': 'log',
            'volume_scale': 'standard',
            'time_features': True
        }
        
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 변환"""
        transformed = data.copy()
        
        if self.transformation_rules['price_scale'] == 'log':
            transformed['close'] = np.log1p(transformed['close'])
            
        if self.transformation_rules['time_features']:
            transformed = self._add_time_features(transformed)
            
        return transformed
