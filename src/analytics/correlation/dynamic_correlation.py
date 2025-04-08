import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import pearsonr

class DynamicCorrelationAnalyzer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.correlation_history = []
        
    def analyze_dynamic_correlation(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """동적 상관관계 분석"""
        symbols = list(price_data.keys())
        returns = {sym: price_data[sym].pct_change() for sym in symbols}
        
        correlation_matrix = pd.DataFrame(
            index=symbols,
            columns=symbols,
            dtype=float
        )
        
        for i in range(len(symbols)):
            for j in range(i, len(symbols)):
                corr = self._calculate_rolling_correlation(
                    returns[symbols[i]], 
                    returns[symbols[j]]
                )
                correlation_matrix.iloc[i, j] = corr
                correlation_matrix.iloc[j, i] = corr
                
        return correlation_matrix
