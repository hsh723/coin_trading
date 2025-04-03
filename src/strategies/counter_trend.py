import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

class CounterTrendStrategy:
    def __init__(self, leverage: float = 40.0):
        self.leverage = leverage
        self.ema_short = 21
        self.ema_long = 60
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표를 계산합니다."""
        # EMA 계산
        df['ema_short'] = EMAIndicator(close=df['close'], window=self.ema_short).ema_indicator()
        df['ema_long'] = EMAIndicator(close=df['close'], window=self.ema_long).ema_indicator()
        
        # RSI 계산
        df['rsi'] = RSIIndicator(close=df['close'], window=self.rsi_period).rsi()
        
        return df
    
    def determine_trend(self, df: pd.DataFrame) -> str:
        """현재 추세를 판단합니다."""
        current_ema_short = df['ema_short'].iloc[-1]
        current_ema_long = df['ema_long'].iloc[-1]
        
        if current_ema_short > current_ema_long:
            return 'up'
        else:
            return 'down'
    
    def check_reversal_opportunity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """역추세 매매 기회를 확인합니다."""
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        trend = self.determine_trend(df)
        
        signal = {
            'position': None,
            'reason': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # 상승 추세에서 과매수 구간 도달 시 숏 진입 기회
        if trend == 'up' and current_rsi > self.rsi_overbought and prev_rsi <= self.rsi_overbought:
            signal['position'] = 'short'
            signal['reason'] = '상승 추세에서 과매수 구간 도달'
            signal['entry_price'] = df['close'].iloc[-1]
            signal['stop_loss'] = signal['entry_price'] * 1.1  # 10% 손절
            signal['take_profit'] = signal['entry_price'] * 0.85  # 15% 익절
            
        # 하락 추세에서 과매도 구간 도달 시 롱 진입 기회
        elif trend == 'down' and current_rsi < self.rsi_oversold and prev_rsi >= self.rsi_oversold:
            signal['position'] = 'long'
            signal['reason'] = '하락 추세에서 과매도 구간 도달'
            signal['entry_price'] = df['close'].iloc[-1]
            signal['stop_loss'] = signal['entry_price'] * 0.9  # 10% 손절
            signal['take_profit'] = signal['entry_price'] * 1.15  # 15% 익절
            
        return signal
    
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호를 생성합니다."""
        df = self.calculate_indicators(df)
        signal = self.check_reversal_opportunity(df)
        
        if signal['position'] is not None:
            signal['timestamp'] = datetime.now()
            signal['trend'] = self.determine_trend(df)
            signal['rsi'] = df['rsi'].iloc[-1]
            
        return signal
    
    def calculate_position_size(self, capital: float, entry_price: float) -> float:
        """포지션 크기를 계산합니다."""
        risk_per_trade = capital * 0.01  # 자본의 1% 리스크
        position_size = (risk_per_trade * self.leverage) / entry_price
        return position_size 