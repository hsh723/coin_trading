import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import talib

class FibonacciStrategy:
    def __init__(self):
        self.levels = [0.382, 0.5, 0.618]  # 주요 피보나치 되돌림 레벨
        self.risk_per_trade = 0.1  # 전체 자본의 10% 리스크
        self.take_profit_ratio = 0.15  # 15% 익절
        self.stop_loss_ratio = 0.1  # 10% 손절
        
    def find_swing_points(self, df: pd.DataFrame) -> Tuple[float, float]:
        """스윙 고점과 저점을 찾습니다."""
        # 20일 기간 동안의 고점과 저점
        high = df['high'].rolling(window=20).max()
        low = df['low'].rolling(window=20).min()
        
        # 최근 고점과 저점
        recent_high = high.iloc[-1]
        recent_low = low.iloc[-1]
        
        return recent_high, recent_low
        
    def calculate_fibonacci_levels(self, high: float, low: float) -> Dict[float, float]:
        """피보나치 되돌림 레벨을 계산합니다."""
        diff = high - low
        levels = {}
        
        for level in self.levels:
            retracement = high - (diff * level)
            levels[level] = retracement
            
        return levels
        
    def check_support_resistance(self, df: pd.DataFrame, levels: Dict[float, float]) -> Dict[str, Any]:
        """지지와 저항 레벨을 확인합니다."""
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # 가장 가까운 피보나치 레벨 찾기
        closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price))
        level_value, price_level = closest_level
        
        # 지지/저항 확인
        is_support = current_price > price_level and prev_price < price_level
        is_resistance = current_price < price_level and prev_price > price_level
        
        return {
            'level': level_value,
            'price': price_level,
            'is_support': is_support,
            'is_resistance': is_resistance
        }
        
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호를 생성합니다."""
        # 스윙 포인트 찾기
        high, low = self.find_swing_points(df)
        
        # 피보나치 레벨 계산
        fib_levels = self.calculate_fibonacci_levels(high, low)
        
        # 지지/저항 확인
        level_info = self.check_support_resistance(df, fib_levels)
        
        signal = {
            'position': None,
            'reason': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        current_price = df['close'].iloc[-1]
        
        # 롱 진입 조건: 38.2% 또는 50% 레벨에서 지지 확인
        if (level_info['is_support'] and 
            (level_info['level'] == 0.382 or level_info['level'] == 0.5)):
            signal['position'] = 'long'
            signal['reason'] = f"피보나치 {level_info['level']*100}% 지지선에서 반등"
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * (1 - self.stop_loss_ratio)
            signal['take_profit'] = current_price * (1 + self.take_profit_ratio)
            
        # 숏 진입 조건: 38.2% 또는 50% 레벨에서 저항 확인
        elif (level_info['is_resistance'] and 
              (level_info['level'] == 0.382 or level_info['level'] == 0.5)):
            signal['position'] = 'short'
            signal['reason'] = f"피보나치 {level_info['level']*100}% 저항선에서 반락"
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * (1 + self.stop_loss_ratio)
            signal['take_profit'] = current_price * (1 - self.take_profit_ratio)
            
        return signal
        
    def check_exit_conditions(self, df: pd.DataFrame, position: str, entry_price: float) -> bool:
        """청산 조건을 확인합니다."""
        current_price = df['close'].iloc[-1]
        
        # 롱 포지션 청산 조건
        if position == 'long':
            # 61.8% 레벨에서 저항 확인
            high, low = self.find_swing_points(df)
            fib_levels = self.calculate_fibonacci_levels(high, low)
            level_618 = fib_levels[0.618]
            
            if current_price >= level_618:
                return True
                
        # 숏 포지션 청산 조건
        elif position == 'short':
            # 61.8% 레벨에서 지지 확인
            high, low = self.find_swing_points(df)
            fib_levels = self.calculate_fibonacci_levels(high, low)
            level_618 = fib_levels[0.618]
            
            if current_price <= level_618:
                return True
                
        return False 