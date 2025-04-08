from typing import List, Dict
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_selection import select_features

class AutoFeatureGenerator:
    def __init__(self, feature_settings: Dict = None):
        self.feature_settings = feature_settings or self._default_settings()
        self.selected_features = []
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """자동 특성 생성 및 선택"""
        # 시계열 특성 추출
        extracted = extract_features(
            data,
            column_id="symbol",
            column_sort="timestamp",
            default_fc_parameters=self.feature_settings
        )
        
        # 중요 특성 선택
        if 'target' in data.columns:
            selected = select_features(extracted, data['target'])
            self.selected_features = selected.columns.tolist()
            return selected
        return extracted
