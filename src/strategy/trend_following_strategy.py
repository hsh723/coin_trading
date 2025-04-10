from typing import Dict
from .base_strategy import BaseStrategy, StrategyResult
import numpy as np

class TrendFollowingStrategy(BaseStrategy):
    async def analyze_market(self, market_data: Dict) -> Dict:
        """시장 분석"""
        return {
            'trend': self._calculate_trend(market_data['close']),
            'momentum': self._calculate_momentum(market_data),
            'volatility': self._calculate_volatility(market_data)
        }
        
    async def generate_signals(self, analysis: Dict) -> StrategyResult:
        """신호 생성"""
        if analysis['trend']['direction'] == 'up' and analysis['momentum'] > 0:
            return StrategyResult(
                signal='buy',
                confidence=min(analysis['trend']['strength'], 1.0),
                params={'trend_strength': analysis['trend']['strength']},
                metadata={'strategy_type': 'trend_following'}
            )
