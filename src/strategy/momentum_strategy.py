from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..analysis.technical import TechnicalAnalyzer

class MomentumStrategy(BaseStrategy):
    def __init__(self, params: Dict):
        super().__init__()
        self.tech_analyzer = TechnicalAnalyzer()
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.rsi_oversold = params.get('rsi_oversold', 30)
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """모멘텀 기반 거래 신호 생성"""
        self.tech_analyzer.set_data(data)
        rsi = self.tech_analyzer.calculate_rsi(self.rsi_period)
        
        signals = {}
        if rsi.iloc[-1] < self.rsi_oversold:
            signals['action'] = 'BUY'
        elif rsi.iloc[-1] > self.rsi_overbought:
            signals['action'] = 'SELL'
        else:
            signals['action'] = 'HOLD'
            
        return signals
