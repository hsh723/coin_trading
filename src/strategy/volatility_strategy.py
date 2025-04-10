from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
from .base_strategy import BaseStrategy, StrategyResult
from ..analysis.technical import TechnicalAnalyzer
import numpy as np

@dataclass
class VolatilitySignal:
    signal_type: str
    target_price: float
    current_volatility: float
    stop_loss: float

class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.vol_window = config.get('volatility_window', 20)
        self.vol_threshold = config.get('volatility_threshold', 2.0)
        self.config = config or {
            'k_value': 0.5,
            'lookback_period': 20,
            'volatility_threshold': 0.02
        }
        
    async def analyze_market(self, market_data: Dict) -> Dict:
        """시장 분석"""
        return {
            'volatility': self._calculate_volatility(market_data),
            'range': self._calculate_daily_range(market_data),
            'breakout_levels': self._calculate_breakout_levels(market_data)
        }
        
    async def generate_signals(self, analysis: Dict) -> Dict:
        k_value = self.config.get('k_value', 0.5)
        target_price = analysis['range']['open'] + (analysis['range']['high'] - analysis['range']['low']) * k_value
        
        if analysis['current_price'] > target_price:
            return StrategyResult(
                signal='buy',
                confidence=self._calculate_signal_strength(analysis),
                params={'target_price': target_price},
                metadata={'strategy_type': 'volatility_breakout'}
            )
            
    async def generate_signal(self, market_data: pd.DataFrame) -> VolatilitySignal:
        """변동성 돌파 신호 생성"""
        daily_range = market_data['high'] - market_data['low']
        volatility = daily_range.std()
        
        target_price = market_data['open'].iloc[-1] + \
                      (daily_range.iloc[-1] * self.config['k_value'])
                      
        return VolatilitySignal(
            signal_type='buy' if market_data['close'].iloc[-1] > target_price else 'hold',
            target_price=target_price,
            current_volatility=volatility,
            stop_loss=target_price * (1 - volatility)
        )
