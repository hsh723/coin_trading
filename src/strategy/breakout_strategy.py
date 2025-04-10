from typing import Dict
import pandas as pd
from .base_strategy import BaseStrategy, StrategyResult
from ..analysis.technical import TechnicalAnalyzer
import numpy as np

class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.lookback_period = params.get('lookback_period', 20)
        
    async def analyze_market(self, market_data: Dict) -> Dict:
        """시장 분석"""
        highs = market_data['high']
        lows = market_data['low']
        
        return {
            'resistance': self._find_resistance_levels(highs),
            'support': self._find_support_levels(lows),
            'volatility': self._calculate_volatility(market_data),
            'volume_confirm': self._check_volume_confirmation(market_data)
        }
        
    async def generate_signals(self, analysis: Dict) -> Dict:
        current_price = analysis['current_price']
        
        if self._is_resistance_breakout(current_price, analysis):
            return StrategyResult(
                signal='buy',
                confidence=self._calculate_breakout_strength(analysis),
                params={'type': 'resistance_break'},
                metadata={'timeframe': self.config['timeframe']}
            )
            
        return StrategyResult(signal='hold', confidence=0.0, params={}, metadata={})
