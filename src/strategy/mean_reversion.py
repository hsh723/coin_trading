from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from src.analysis.indicators.technical import TechnicalIndicators

class MeanReversionStrategy(BaseStrategy):
    """평균 회귀 기반 거래 전략"""
    
    def __init__(self, 
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        super().__init__()
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self._state = {}
        
    def initialize(self, data: pd.DataFrame) -> None:
        """전략 초기화"""
        super().initialize(data)
        self._state = {
            'initialized': True,
            'last_signal': None,
            'position': None
        }
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시장 분석"""
        # 볼린저 밴드 계산
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
            data['close'],
            self.bb_period,
            self.bb_std
        )
        
        # RSI 계산
        rsi = TechnicalIndicators.calculate_rsi(data['close'], self.rsi_period)
        
        return {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'rsi': rsi
        }
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래 신호 생성"""
        super().generate_signals(data)
        analysis = self.analyze(data)
        latest = {
            'price': data['close'].iloc[-1],
            'bb_upper': analysis['bb_upper'].iloc[-1],
            'bb_middle': analysis['bb_middle'].iloc[-1],
            'bb_lower': analysis['bb_lower'].iloc[-1],
            'rsi': analysis['rsi'].iloc[-1]
        }
        
        # 매수 조건: 가격이 하단 밴드 아래이고 RSI가 과매도
        buy_signal = (
            latest['price'] < latest['bb_lower'] and
            latest['rsi'] < self.rsi_oversold
        )
        
        # 매도 조건: 가격이 상단 밴드 위이고 RSI가 과매수
        sell_signal = (
            latest['price'] > latest['bb_upper'] and
            latest['rsi'] > self.rsi_overbought
        )
        
        return {
            'buy': buy_signal,
            'sell': sell_signal,
            'analysis': latest
        }
        
    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래 실행"""
        signals = self.generate_signals(data)
        
        if signals['buy'] and self._state['position'] is None:
            self._state['position'] = 'long'
            return {'action': 'buy', 'price': data['close'].iloc[-1]}
        elif signals['sell'] and self._state['position'] == 'long':
            self._state['position'] = None
            return {'action': 'sell', 'price': data['close'].iloc[-1]}
            
        return {'action': 'hold'}
        
    def update(self, data: pd.DataFrame) -> None:
        """전략 상태 업데이트"""
        self._state['last_signal'] = self.generate_signals(data)
        
    def get_state(self) -> Dict[str, Any]:
        """전략 상태 조회"""
        return self._state.copy()
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """전략 상태 설정"""
        self._state = state.copy() 