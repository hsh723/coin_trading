from typing import Dict, List
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer

class MultiTimeframeStrategy(BaseStrategy):
    def __init__(self, timeframes: List[str], params: Dict):
        super().__init__()
        self.timeframes = timeframes
        self.analyzers = {tf: TechnicalAnalyzer() for tf in timeframes}
        self.weights = params.get('weights', {
            '1h': 0.3,
            '4h': 0.3,
            '1d': 0.4
        })
        
    async def analyze_timeframes(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """각 타임프레임별 분석 수행"""
        signals = {}
        for timeframe in self.timeframes:
            self.analyzers[timeframe].set_data(data[timeframe])
            signals[timeframe] = await self._analyze_single_timeframe(timeframe)
            
        return self._combine_signals(signals)
        
    def _combine_signals(self, signals: Dict[str, str]) -> Dict[str, str]:
        """타임프레임별 신호 통합"""
        signal_score = 0
        for timeframe, signal in signals.items():
            if signal == 'BUY':
                signal_score += self.weights[timeframe]
            elif signal == 'SELL':
                signal_score -= self.weights[timeframe]
                
        if signal_score > 0.2:
            return {'action': 'BUY'}
        elif signal_score < -0.2:
            return {'action': 'SELL'}
        return {'action': 'HOLD'}
