import numpy as np
from typing import Dict
from ..analysis.technical import TechnicalAnalyzer

class DynamicRiskManager:
    def __init__(self, config: Dict):
        self.max_position_size = config.get('max_position_size', 0.1)
        self.base_risk_per_trade = config.get('base_risk_per_trade', 0.02)
        self.tech_analyzer = TechnicalAnalyzer()
        
    def calculate_position_size(self, data: pd.DataFrame, signal_strength: float) -> float:
        """변동성 기반 포지션 크기 계산"""
        atr = self.tech_analyzer.calculate_atr(data)
        volatility_factor = self._calculate_volatility_factor(atr)
        return self.base_risk_per_trade * signal_strength * volatility_factor
        
    def _calculate_volatility_factor(self, atr: float) -> float:
        """변동성 기반 리스크 조정"""
        return np.clip(1.0 / atr, 0.5, 2.0)
