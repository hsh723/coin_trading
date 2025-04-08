import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from typing import Dict, List, Tuple

class CointegrationAnalyzer:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def find_cointegrated_pairs(self, price_data: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """공적분 페어 탐색"""
        n = len(price_data.columns)
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                score, pvalue, _ = coint(price_data.iloc[:, i], price_data.iloc[:, j])
                if pvalue < self.significance_level:
                    pairs.append((
                        price_data.columns[i],
                        price_data.columns[j],
                        pvalue
                    ))
        return pairs
