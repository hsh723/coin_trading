import pandas as pd
import numpy as np
from typing import Dict

class DataNormalizer:
    def __init__(self, normalization_method: str = 'zscore'):
        self.normalization_method = normalization_method
        self.stats_cache = {}
        
    async def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정규화"""
        if self.normalization_method == 'zscore':
            return self._apply_zscore_normalization(data)
        elif self.normalization_method == 'minmax':
            return self._apply_minmax_normalization(data)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
