from typing import Dict, List
import pandas as pd
from .base import BaseStrategy
from ..analysis.correlation_analyzer import CorrelationAnalyzer

class MultiMarketStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.corr_analyzer = CorrelationAnalyzer()
        self.min_markets = config.get('min_markets', 3)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """다중 시장 신호 생성"""
        # 시장간 상관관계 분석
        correlation_matrix = self.corr_analyzer.analyze_markets(market_data)
        uncorrelated_markets = self._select_uncorrelated_markets(correlation_matrix)
        
        signals = {}
        for market in uncorrelated_markets:
            market_signal = await self._analyze_single_market(market_data[market])
            signals[market] = market_signal
            
        return self._combine_market_signals(signals)
