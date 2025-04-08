from typing import Dict, Any, Optional
import pandas as pd
from src.strategy.base_strategy import BaseStrategy
from src.analysis.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

class MomentumStrategy(BaseStrategy):
    """모멘텀 전략 클래스"""
    
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
        
    def initialize(self, data: pd.DataFrame) -> None:
        """전략 초기화"""
        self._state = {
            'last_signal': None,
            'position': 0,
            'entry_price': 0.0
        }
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시장 분석"""
        rsi = TechnicalIndicators.calculate_rsi(data['close'], self.rsi_period)
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
        """매매 신호 생성"""
        analysis = self.analyze(data)
        latest = data.iloc[-1]
        
        buy_signal = (
            (analysis['rsi'].iloc[-1] > self.rsi_oversold) &
            (analysis['macd'].iloc[-1] > analysis['signal'].iloc[-1]) &
            (analysis['macd'].iloc[-2] <= analysis['signal'].iloc[-2])
        )
        
        sell_signal = (
            (analysis['rsi'].iloc[-1] < self.rsi_overbought) &
            (analysis['macd'].iloc[-1] < analysis['signal'].iloc[-1]) &
            (analysis['macd'].iloc[-2] >= analysis['signal'].iloc[-2])
        )
        
        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'price': latest['close'],
            'analysis': {
                'rsi': analysis['rsi'].iloc[-1],
                'macd': analysis['macd'].iloc[-1],
                'signal': analysis['signal'].iloc[-1]
            }
        }
        
    def execute(self, data: pd.DataFrame, position: Optional[float] = None) -> Dict[str, Any]:
        """매매 실행"""
        signals = self.generate_signals(data)
        
        if position is None or position == 0:
            if signals['buy_signal']:
                return {'action': 'buy', 'amount': 1.0}
        else:
            if signals['sell_signal']:
                return {'action': 'sell', 'amount': position}
                
        return {'action': 'hold', 'amount': 0.0}
        
    def update(self, data: pd.DataFrame) -> None:
        """전략 상태 업데이트"""
        signals = self.generate_signals(data)
        self._state['last_signal'] = signals
        
    def get_state(self) -> Dict[str, Any]:
        """전략 상태 반환"""
        return self._state
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """전략 상태 설정"""
        self._state = state 