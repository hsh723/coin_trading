import numpy as np
import pandas as pd
from typing import Dict, Any
import talib

class BollingerStrategy:
    def __init__(self):
        self.ma_200 = 200  # 200일 이동평균
        self.risk_per_trade = 0.1  # 전체 자본의 10% 리스크
        self.take_profit_ratio = 0.15  # 15% 익절
        self.stop_loss_ratio = 0.15  # 15% 손절
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """이동평균선을 계산합니다."""
        # 200일 이동평균선 계산
        df['ma_200'] = talib.SMA(df['close'], timeperiod=self.ma_200)
        
        # 고점/저점 계산
        df['high_peak'] = df['high'].rolling(window=5, center=True).max()
        df['low_peak'] = df['low'].rolling(window=5, center=True).min()
        
        return df
        
    def check_trend(self, df: pd.DataFrame) -> str:
        """추세를 판단합니다."""
        current_price = df['close'].iloc[-1]
        ma_200 = df['ma_200'].iloc[-1]
        
        if current_price > ma_200:
            return 'up'  # 상승추세
        else:
            return 'down'  # 하락추세
            
    def find_abc_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """ABC 포인트를 찾습니다."""
        # 최근 100개의 캔들에서 고점/저점 찾기
        recent_df = df.tail(100)
        
        # A 포인트 (신고가)
        a_point = recent_df['high'].max()
        a_index = recent_df['high'].idxmax()
        
        # B 포인트 (A 이후의 저점)
        after_a = recent_df.loc[a_index:]
        b_point = after_a['low'].min()
        b_index = after_a['low'].idxmin()
        
        # C 포인트 (B 이후의 고점)
        after_b = recent_df.loc[b_index:]
        c_point = after_b['high'].max()
        c_index = after_b['high'].idxmax()
        
        # D 포인트 (C 이후의 저점)
        after_c = recent_df.loc[c_index:]
        d_point = after_c['low'].min()
        
        return {
            'a': a_point,
            'b': b_point,
            'c': c_point,
            'd': d_point
        }
        
    def check_abc_pattern(self, df: pd.DataFrame, points: Dict[str, float]) -> bool:
        """ABC 패턴이 유효한지 확인합니다."""
        # C가 A를 넘지 않아야 함
        if points['c'] > points['a']:
            return False
            
        # D가 B를 넘지 않아야 함
        if points['d'] < points['b']:
            return False
            
        return True
        
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호를 생성합니다."""
        df = self.calculate_indicators(df)
        trend = self.check_trend(df)
        points = self.find_abc_points(df)
        
        signal = {
            'position': None,
            'reason': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        current_price = df['close'].iloc[-1]
        
        # ABC 패턴이 유효한 경우에만 진입
        if self.check_abc_pattern(df, points):
            # 상승추세 + D 포인트 근처 = 롱 진입
            if trend == 'up' and abs(current_price - points['d']) / points['d'] < 0.01:
                signal['position'] = 'long'
                signal['reason'] = "상승추세 + ABC 패턴 D 포인트"
                signal['entry_price'] = current_price
                signal['stop_loss'] = current_price * (1 - self.stop_loss_ratio)
                signal['take_profit'] = current_price * (1 + self.take_profit_ratio)
                
            # 하락추세 + D 포인트 근처 = 숏 진입
            elif trend == 'down' and abs(current_price - points['d']) / points['d'] < 0.01:
                signal['position'] = 'short'
                signal['reason'] = "하락추세 + ABC 패턴 D 포인트"
                signal['entry_price'] = current_price
                signal['stop_loss'] = current_price * (1 + self.stop_loss_ratio)
                signal['take_profit'] = current_price * (1 - self.take_profit_ratio)
                
        return signal 