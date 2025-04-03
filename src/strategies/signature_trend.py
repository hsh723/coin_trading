import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import talib

class SignatureTrendStrategy:
    def __init__(self, leverage: float = 40.0):
        self.leverage = leverage
        self.ema_short = 21
        self.ema_long = 60
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표를 계산합니다."""
        # EMA 계산
        df['ema_short'] = talib.EMA(df['close'], timeperiod=self.ema_short)
        df['ema_long'] = talib.EMA(df['close'], timeperiod=self.ema_long)
        
        # RSI 계산
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # MACD 계산
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        
        # 볼린저 밴드 계산
        df['upper'], df['middle'], df['lower'] = talib.BBANDS(
            df['close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        
        return df
    
    def determine_trend(self, df: pd.DataFrame) -> str:
        """현재 추세를 판단합니다."""
        current_ema_short = df['ema_short'].iloc[-1]
        current_ema_long = df['ema_long'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        
        # EMA와 MACD를 함께 사용하여 추세 판단
        if (current_ema_short > current_ema_long and 
            current_macd > current_macd_signal):
            return 'up'
        elif (current_ema_short < current_ema_long and 
              current_macd < current_macd_signal):
            return 'down'
        else:
            return 'neutral'
    
    def check_entry_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """진입 조건을 확인합니다."""
        current_rsi = df['rsi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        current_lower = df['lower'].iloc[-1]
        current_upper = df['upper'].iloc[-1]
        trend = self.determine_trend(df)
        
        signal = {
            'position': None,
            'reason': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # 롱 진입 조건
        if (trend == 'up' and 
            current_rsi < self.rsi_oversold and 
            current_price <= current_lower):
            signal['position'] = 'long'
            signal['reason'] = '하락 반등 기대'
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * 0.9  # 10% 손절
            signal['take_profit'] = current_price * 1.15  # 15% 익절
            
        # 숏 진입 조건
        elif (trend == 'down' and 
              current_rsi > self.rsi_overbought and 
              current_price >= current_upper):
            signal['position'] = 'short'
            signal['reason'] = '상승 반락 기대'
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * 1.1  # 10% 손절
            signal['take_profit'] = current_price * 0.85  # 15% 익절
            
        return signal
    
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호를 생성합니다."""
        df = self.calculate_indicators(df)
        signal = self.check_entry_conditions(df)
        
        if signal['position'] is not None:
            signal['timestamp'] = datetime.now()
            signal['trend'] = self.determine_trend(df)
            signal['rsi'] = df['rsi'].iloc[-1]
            signal['macd'] = df['macd'].iloc[-1]
            
        return signal
    
    def calculate_position_size(self, capital: float, entry_price: float) -> float:
        """포지션 크기를 계산합니다."""
        risk_per_trade = capital * 0.01  # 자본의 1% 리스크
        position_size = (risk_per_trade * self.leverage) / entry_price
        return position_size 