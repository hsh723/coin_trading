from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer
from ..analysis.machine_learning import MLAnalyzer

@dataclass
class AdaptiveParameters:
    timeframe: str
    volatility_factor: float
    momentum_threshold: float
    trend_strength: float

class AdaptiveStrategy(BaseStrategy):
    def __init__(self, config: Dict = None):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.ml_analyzer = MLAnalyzer()
        self.volatility_threshold = config.get('volatility_threshold', 0.02) if config else 0.02
        self.config = config or {
            'adaptation_period': 24,  # hours
            'min_samples': 100,
            'learning_rate': 0.01
        }
        self.parameters = {}
        
    def analyze_market_condition(self, data: pd.DataFrame) -> str:
        """시장 상황 분석"""
        atr = self.tech_analyzer.calculate_atr(data)
        trend = self.tech_analyzer.identify_trend()
        
        if atr[-1] > self.volatility_threshold:
            return 'HIGH_VOLATILITY'
        else:
            return 'LOW_VOLATILITY'
            
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """시장 상황에 따른 신호 생성"""
        market_condition = self.analyze_market_condition(data)
        if market_condition == 'HIGH_VOLATILITY':
            return self._generate_trend_signals(data)
        else:
            return self._generate_mean_reversion_signals(data)
            
    async def adapt_strategy(self, market_data: pd.DataFrame) -> AdaptiveParameters:
        """시장 상황에 따른 전략 파라미터 조정"""
        volatility = self._calculate_current_volatility(market_data)
        trend = self._detect_market_regime(market_data)
        
        return AdaptiveParameters(
            timeframe=self._select_optimal_timeframe(volatility),
            volatility_factor=self._adjust_volatility_factor(volatility),
            momentum_threshold=self._calculate_momentum_threshold(trend),
            trend_strength=self._measure_trend_strength(market_data)
        )
