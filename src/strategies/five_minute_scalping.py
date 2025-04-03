import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import talib

class FiveMinuteScalping:
    def __init__(self, leverage: float = 40.0):
        self.leverage = leverage
        self.ema_short = 21
        self.ema_long = 60
        self.stoch_k = 14
        self.stoch_d = 3
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표를 계산합니다."""
        # EMA 계산
        df['ema_short'] = talib.EMA(df['close'], timeperiod=self.ema_short)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=self.ema_long)
        
        # 스토캐스틱 계산
        df['slowk'], df['slowd'] = talib.STOCH(
            df['high'], 
            df['low'], 
            df['close'],
            fastk_period=self.stoch_k,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        
        return df
    
    def determine_trend(self, df: pd.DataFrame) -> str:
        """현재 추세를 판단합니다."""
        current_ema_short = df['ema_short'].iloc[-1]
        current_ema_long = df['ema_long'].iloc[-1]
        
        if current_ema_short > current_ema_long:
            return 'up'
        else:
            return 'down'
    
    def check_stochastic_signal(self, df: pd.DataFrame) -> str:
        """스토캐스틱 신호를 확인합니다."""
        current_k = df['slowk'].iloc[-1]
        current_d = df['slowd'].iloc[-1]
        
        if current_k > 80 and current_d > 80:
            return 'overbought'
        elif current_k < 20 and current_d < 20:
            return 'oversold'
        else:
            return 'neutral'
    
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호를 생성합니다."""
        df = self.calculate_indicators(df)
        trend = self.determine_trend(df)
        stoch_signal = self.check_stochastic_signal(df)
        
        signal = {
            'timestamp': datetime.now(),
            'trend': trend,
            'stochastic': stoch_signal,
            'position': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # 매매 신호 생성
        if trend == 'up' and stoch_signal == 'oversold':
            signal['position'] = 'long'
            signal['entry_price'] = df['close'].iloc[-1]
            signal['stop_loss'] = signal['entry_price'] * 0.9  # 10% 손절
            signal['take_profit'] = signal['entry_price'] * 1.15  # 15% 익절
            
        elif trend == 'down' and stoch_signal == 'overbought':
            signal['position'] = 'short'
            signal['entry_price'] = df['close'].iloc[-1]
            signal['stop_loss'] = signal['entry_price'] * 1.1  # 10% 손절
            signal['take_profit'] = signal['entry_price'] * 0.85  # 15% 익절
            
        return signal
    
    def calculate_position_size(self, capital: float, entry_price: float) -> float:
        """포지션 크기를 계산합니다."""
        risk_per_trade = capital * 0.01  # 자본의 1% 리스크
        position_size = (risk_per_trade * self.leverage) / entry_price
        return position_size 