import pandas as pd
import numpy as np
from typing import List, Dict
from tsfresh import extract_features
from tsfresh.feature_selection import select_features

class TimeSeriesFeatureGenerator:
    def __init__(self, feature_settings: Dict = None):
        self.feature_settings = feature_settings or self._default_settings()
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """시계열 특성 생성"""
        extracted_features = extract_features(
            data,
            column_id="symbol",
            column_sort="timestamp",
            default_fc_parameters=self.feature_settings
        )
        return self._select_relevant_features(extracted_features)
