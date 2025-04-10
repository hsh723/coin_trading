from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
import statsmodels.api as sm
from .base import BaseStrategy
from ..analysis.correlation_analyzer import CorrelationAnalyzer

@dataclass
class StatArbSignal:
    pair_symbols: tuple
    zscore: float
    hedge_ratio: float
    entry_threshold: float
    exit_threshold: float

class StatisticalArbitrageStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {
            'lookback_period': 100,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'min_half_life': 12
        }
        
    async def find_pairs(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """공적분 관계가 있는 페어 찾기"""
        pairs = []
        symbols = list(market_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if await self._test_cointegration(
                    market_data[symbols[i]]['close'],
                    market_data[symbols[j]]['close']
                ):
                    pairs.append({
                        'symbol1': symbols[i],
                        'symbol2': symbols[j]
                    })
        
        return pairs

    async def find_opportunities(self, pair_data: Dict[str, pd.DataFrame]) -> List[StatArbSignal]:
        """통계적 차익거래 기회 탐색"""
        opportunities = []
        for pair in self._generate_pairs(pair_data):
            if self._is_cointegrated(pair_data[pair[0]], pair_data[pair[1]]):
                signal = self._calculate_spread_signal(pair, pair_data)
                if abs(signal.zscore) > self.config['entry_threshold']:
                    opportunities.append(signal)
                    
        return opportunities
