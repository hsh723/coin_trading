from typing import Dict, List
import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..analysis.correlation_analyzer import CorrelationAnalyzer

class StatisticalArbitrageStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.lookback_period = config.get('lookback_period', 100)
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)
        
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
