import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class FlightStrategy:
    def __init__(self, leverage: float = 50.0):
        self.leverage = leverage
        self.risk_per_trade = 0.1  # 전체 자본의 10% 리스크
        self.take_profit_ratio = 0.2  # 20% 익절
        self.stop_loss_ratio = 0.1  # 10% 손절
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표를 계산합니다."""
        # 거래량 이동평균
        df['volume_ma'] = SMAIndicator(close=df['volume'], window=20).sma_indicator()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # 볼린저 밴드
        bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['upper'] = bollinger.bollinger_hband()
        df['middle'] = bollinger.bollinger_mavg()
        df['lower'] = bollinger.bollinger_lband()
        
        # MACD
        macd = MACD(
            close=df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        return df
    
    def check_volume_divergence(self, df: pd.DataFrame) -> Dict[str, bool]:
        """거래량 다이버전스를 확인합니다."""
        # 상승 다이버전스: 가격은 상승, 거래량은 감소
        price_up = df['close'].iloc[-1] > df['close'].iloc[-2]
        volume_down = df['volume'].iloc[-1] < df['volume'].iloc[-2]
        bullish_divergence = price_up and volume_down
        
        # 하락 다이버전스: 가격은 하락, 거래량은 감소
        price_down = df['close'].iloc[-1] < df['close'].iloc[-2]
        volume_down = df['volume'].iloc[-1] < df['volume'].iloc[-2]
        bearish_divergence = price_down and volume_down
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def check_fractal_pattern(self, df: pd.DataFrame) -> Dict[str, bool]:
        """프랙탈 패턴을 확인합니다."""
        # 상승 프랙탈: 이전 고점보다 현재 고점이 높고, 조정 기간이 짧아짐
        current_high = df['high'].iloc[-1]
        prev_high = df['high'].iloc[-2]
        correction_period = len(df[df['close'] < df['close'].shift(1)])
        
        bullish_fractal = (current_high > prev_high and 
                          correction_period < len(df) * 0.1)
        
        # 하락 프랙탈: 이전 저점보다 현재 저점이 낮고, 반등 기간이 짧아짐
        current_low = df['low'].iloc[-1]
        prev_low = df['low'].iloc[-2]
        bounce_period = len(df[df['close'] > df['close'].shift(1)])
        
        bearish_fractal = (current_low < prev_low and 
                          bounce_period < len(df) * 0.1)
        
        return {
            'bullish_fractal': bullish_fractal,
            'bearish_fractal': bearish_fractal
        }
    
    def check_bounce_opportunity(self, df: pd.DataFrame) -> Dict[str, bool]:
        """반등 기회를 확인합니다."""
        # 급격한 하락 후 반등 기회
        sharp_drop = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] < -0.05
        volume_spike = df['volume'].iloc[-1] > df['volume_ma'].iloc[-1] * 1.5
        bounce_opportunity = sharp_drop and volume_spike
        
        # 급격한 상승 후 반락 기회
        sharp_rise = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] > 0.05
        drop_opportunity = sharp_rise and volume_spike
        
        return {
            'bounce_opportunity': bounce_opportunity,
            'drop_opportunity': drop_opportunity
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호를 생성합니다."""
        df = self.calculate_indicators(df)
        volume_divergence = self.check_volume_divergence(df)
        fractal_pattern = self.check_fractal_pattern(df)
        bounce_opportunity = self.check_bounce_opportunity(df)
        
        signal = {
            'position': None,
            'reason': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        current_price = df['close'].iloc[-1]
        
        # 롱 진입 조건
        if (volume_divergence['bullish_divergence'] or 
            fractal_pattern['bullish_fractal'] or 
            bounce_opportunity['bounce_opportunity']):
            signal['position'] = 'long'
            signal['reason'] = '상승 반등 기회'
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * (1 - self.stop_loss_ratio)
            signal['take_profit'] = current_price * (1 + self.take_profit_ratio)
            
        # 숏 진입 조건
        elif (volume_divergence['bearish_divergence'] or 
              fractal_pattern['bearish_fractal'] or 
              bounce_opportunity['drop_opportunity']):
            signal['position'] = 'short'
            signal['reason'] = '하락 반락 기회'
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * (1 + self.stop_loss_ratio)
            signal['take_profit'] = current_price * (1 - self.take_profit_ratio)
            
        return signal
    
    def calculate_position_size(self, capital: float, entry_price: float) -> float:
        """포지션 크기를 계산합니다."""
        risk_amount = capital * self.risk_per_trade
        position_size = (risk_amount * self.leverage) / entry_price
        return position_size 