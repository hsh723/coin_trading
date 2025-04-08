from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from src.analysis.indicators.technical import TechnicalIndicators

class MomentumStrategy(BaseStrategy):
    """모멘텀 기반 거래 전략"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
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
        # RSI 계산
        rsi = TechnicalIndicators.calculate_rsi(data['close'], self.rsi_period)
        
        # MACD 계산
        macd, signal = TechnicalIndicators.calculate_macd(
            data['close'],
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        
        return {
            'rsi': rsi,
            'macd': macd,
            'signal': signal
        }
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래 신호 생성"""
        super().generate_signals(data)
        analysis = self.analyze(data)
        latest = {
            'price': data['close'].iloc[-1],
            'rsi': analysis['rsi'].iloc[-1],
            'macd': analysis['macd'].iloc[-1],
            'signal': analysis['signal'].iloc[-1]
        }
        
        # 매수 조건: RSI가 과매도 구간을 벗어나고 MACD가 시그널 라인을 상향 돌파
        buy_signal = (
            latest['rsi'] > self.rsi_oversold and
            latest['macd'] > latest['signal'] and
            analysis['macd'].iloc[-2] <= analysis['signal'].iloc[-2]
        )
        
        # 매도 조건: RSI가 과매수 구간을 벗어나고 MACD가 시그널 라인을 하향 돌파
        sell_signal = (
            latest['rsi'] < self.rsi_overbought and
            latest['macd'] < latest['signal'] and
            analysis['macd'].iloc[-2] >= analysis['signal'].iloc[-2]
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