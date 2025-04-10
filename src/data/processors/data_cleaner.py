import pandas as pd
import numpy as np
from typing import Dict

class DataCleaner:
    def __init__(self):
        self.anomaly_threshold = 3.0  # 표준편차 기준
        
    def clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제"""
        # 결측치 처리
        data = self._handle_missing_values(data)
        
        # 이상치 처리
        data = self._remove_outliers(data)
        
        # 중복 제거
        data = self._remove_duplicates(data)
        
        return data
