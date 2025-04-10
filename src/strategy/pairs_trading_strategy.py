from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from scipy import stats

@dataclass
class PairsSignal:
    pair_id: str
    spread: float
    zscore: float
    hedge_ratio: float
    entry_signal: str
    confidence: float

class PairsTrader:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'zscore_threshold': 2.0,
            'lookback_period': 100,
            'min_cointegration': 0.05
        }
        
    async def analyze_pair(self, pair_data: Dict[str, pd.Series]) -> PairsSignal:
        """페어 분석 및 신호 생성"""
        spread = self._calculate_spread(pair_data)
        zscore = self._calculate_zscore(spread)
        hedge_ratio = self._calculate_hedge_ratio(pair_data)
        
        return PairsSignal(
            pair_id=f"{list(pair_data.keys())[0]}_{list(pair_data.keys())[1]}",
            spread=spread[-1],
            zscore=zscore[-1],
            hedge_ratio=hedge_ratio,
            entry_signal=self._generate_signal(zscore[-1]),
            confidence=self._calculate_confidence(zscore[-1])
        )
