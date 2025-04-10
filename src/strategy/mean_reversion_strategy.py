from typing import Dict
from .base_strategy import BaseStrategy, StrategyResult
import pandas as pd
import numpy as np
from ..analysis.technical import TechnicalAnalyzer

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.lookback_period = params.get('lookback_period', 20)
        self.std_dev_threshold = params.get('std_dev_threshold', 2.0)
        
    async def generate_signal(self, market_data: Dict) -> StrategyResult:
        """평균 회귀 전략 신호 생성"""
        zscore = self._calculate_zscore(market_data)
        
        if zscore > self.config['zscore_threshold']:
            signal = 'sell'
        elif zscore < -self.config['zscore_threshold']:
            signal = 'buy'
        else:
            signal = 'hold'
            
        return StrategyResult(
            signal=signal,
            confidence=abs(zscore),
            params={'zscore': zscore},
            metadata={'timeframe': '1h'}
        )
