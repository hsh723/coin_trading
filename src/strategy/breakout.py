from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from src.analysis.indicators.technical import TechnicalIndicators

class BreakoutStrategy(BaseStrategy):
    """돌파 기반 거래 전략"""
    
    def __init__(self, 
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 min_volume: float = 1000.0):
        super().__init__()
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_volume = min_volume
        self._state = {}
        
    def initialize(self, data: pd.DataFrame) -> None:
        """전략 초기화"""
        super().initialize(data)
        self._state = {
            'initialized': True,
            'last_signal': None,
            'position': None,
            'stop_loss': None,
            'take_profit': None
        }
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시장 분석"""
        # ATR 계산
        atr = TechnicalIndicators.calculate_atr(
            data['high'],
            data['low'],
            data['close'],
            self.atr_period
        )
        
        # 볼륨 분석
        volume_ma = data['volume'].rolling(window=20).mean()
        
        return {
            'atr': atr,
            'volume_ma': volume_ma,
            'high': data['high'],
            'low': data['low'],
            'close': data['close']
        }
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래 신호 생성"""
        super().generate_signals(data)
        analysis = self.analyze(data)
        latest = {
            'price': data['close'].iloc[-1],
            'high': data['high'].iloc[-1],
            'low': data['low'].iloc[-1],
            'volume': data['volume'].iloc[-1],
            'atr': analysis['atr'].iloc[-1],
            'volume_ma': analysis['volume_ma'].iloc[-1]
        }
        
        # 매수 조건: 상단 돌파 및 볼륨 증가
        buy_signal = (
            latest['price'] > latest['high'] and
            latest['volume'] > latest['volume_ma'] and
            latest['volume'] > self.min_volume
        )
        
        # 매도 조건: 하단 돌파
        sell_signal = latest['price'] < latest['low']
        
        return {
            'buy': buy_signal,
            'sell': sell_signal,
            'analysis': latest
        }
        
    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래 실행"""
        signals = self.generate_signals(data)
        latest = signals['analysis']
        
        if signals['buy'] and self._state['position'] is None:
            self._state['position'] = 'long'
            self._state['stop_loss'] = latest['price'] - (latest['atr'] * self.atr_multiplier)
            self._state['take_profit'] = latest['price'] + (latest['atr'] * self.atr_multiplier)
            return {'action': 'buy', 'price': latest['price']}
        elif signals['sell'] or latest['price'] < self._state['stop_loss']:
            self._state['position'] = None
            self._state['stop_loss'] = None
            self._state['take_profit'] = None
            return {'action': 'sell', 'price': latest['price']}
            
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