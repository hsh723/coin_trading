from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """모멘텀 기반 거래 전략"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
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
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """거래 신호 계산"""
        # RSI 계산
        from src.analysis.indicators.technical import TechnicalIndicators
        rsi = TechnicalIndicators.calculate_rsi(data['close'], self.rsi_period)
        
        # MACD 계산
        macd = TechnicalIndicators.calculate_macd(
            data['close'],
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        
        # 매수/매도 신호 생성
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = rsi
        signals['macd'] = macd['macd']
        signals['signal'] = macd['signal']
        
        # 매수 조건: RSI가 과매도 구간이고 MACD가 시그널선을 상향 돌파
        signals['buy_signal'] = (
            (signals['rsi'] < self.rsi_oversold) &
            (signals['macd'] > signals['signal']) &
            (signals['macd'].shift(1) <= signals['signal'].shift(1))
        )
        
        # 매도 조건: RSI가 과매수 구간이고 MACD가 시그널선을 하향 돌파
        signals['sell_signal'] = (
            (signals['rsi'] > self.rsi_overbought) &
            (signals['macd'] < signals['signal']) &
            (signals['macd'].shift(1) >= signals['signal'].shift(1))
        )
        
        return signals
    
    def execute_trade(self, 
                     data: pd.DataFrame,
                     position: Optional[float] = None,
                     balance: float = 0.0) -> Dict[str, float]:
        """거래 실행"""
        signals = self.calculate_signals(data)
        latest = signals.iloc[-1]
        
        if position is None or position == 0:
            if latest['buy_signal']:
                return {'action': 'buy', 'amount': balance}
        else:
            if latest['sell_signal']:
                return {'action': 'sell', 'amount': position}
                
        return {'action': 'hold', 'amount': 0.0} 