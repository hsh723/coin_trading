from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer
from ..analysis.machine_learning import MLAnalyzer

class AdaptiveStrategy(BaseStrategy):
    def __init__(self, config: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.ml_analyzer = MLAnalyzer()
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        
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
