from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class CorrelationAnalysis:
    correlation_matrix: pd.DataFrame
    high_correlation_pairs: List[Dict]
    stable_relationships: List[Dict]
    regime_changes: List[Dict]

class StrategyCorrelationAnalyzer:
    def __init__(self, correlation_config: Dict = None):
        self.config = correlation_config or {
            'min_correlation': 0.7,
            'window_size': 30,
            'stable_threshold': 0.1
        }
        
    async def analyze_correlations(self, 
                                 price_data: Dict[str, pd.DataFrame]) -> CorrelationAnalysis:
        """자산 간 상관관계 분석"""
        # 모든 자산의 수익률 계산
        returns = self._calculate_returns(price_data)
        
        # 상관관계 행렬 계산
        correlation_matrix = returns.corr()
        
        return CorrelationAnalysis(
            correlation_matrix=correlation_matrix,
            high_correlation_pairs=self._find_high_correlations(correlation_matrix),
            stable_relationships=self._find_stable_relationships(returns),
            regime_changes=self._detect_regime_changes(returns)
        )
