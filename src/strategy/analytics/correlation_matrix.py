import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class CorrelationAnalysis:
    matrix: pd.DataFrame
    stable_pairs: List[tuple]
    risk_clusters: Dict[str, List[str]]
    regime_changes: List[Dict]

class CorrelationMatrix:
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        
    async def analyze_correlations(self, price_data: Dict[str, pd.Series]) -> CorrelationAnalysis:
        df = pd.DataFrame(price_data)
        correlation = df.corr()
        
        return CorrelationAnalysis(
            matrix=correlation,
            stable_pairs=self._find_stable_pairs(correlation),
            risk_clusters=self._identify_clusters(correlation),
            regime_changes=self._detect_regime_changes(df)
        )
