import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FibonacciLevel(Enum):
    LEVEL_236 = 0.236
    LEVEL_382 = 0.382
    LEVEL_500 = 0.500
    LEVEL_618 = 0.618
    LEVEL_786 = 0.786

@dataclass
class FibonacciLevels:
    levels: Dict[FibonacciLevel, float]
    high: float
    low: float
    is_uptrend: bool
    current_price: float
    nearest_level: Optional[FibonacciLevel]
    distance_to_level: float

@dataclass
class FibonacciSignal:
    type: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0.0 ~ 1.0
    level: Optional[FibonacciLevel]
    price: float
    reason: str

class FibonacciIndicator:
    def __init__(self):
        self.levels = [
            FibonacciLevel.LEVEL_236,
            FibonacciLevel.LEVEL_382,
            FibonacciLevel.LEVEL_500,
            FibonacciLevel.LEVEL_618,
            FibonacciLevel.LEVEL_786
        ]
        self.level_weights = {
            FibonacciLevel.LEVEL_382: 0.3,
            FibonacciLevel.LEVEL_500: 0.3,
            FibonacciLevel.LEVEL_618: 0.4
        }
        self.price_threshold = 0.001  # 가격이 레벨에 얼마나 가까워야 하는지 (0.1%)

    def calculate_fibonacci_levels(
        self,
        high: float,
        low: float,
        is_uptrend: bool = True
    ) -> FibonacciLevels:
        """
        피보나치 레벨 계산
        
        Args:
            high (float): 고점
            low (float): 저점
            is_uptrend (bool): 상승 추세 여부
            
        Returns:
            FibonacciLevels: 계산된 피보나치 레벨 정보
        """
        levels = {}
        current_price = high if is_uptrend else low
        
        # 레벨 계산
        for level in self.levels:
            if is_uptrend:
                price = high - (high - low) * level.value
            else:
                price = low + (high - low) * level.value
            levels[level] = price
            
        # 가장 가까운 레벨 찾기
        nearest_level = min(
            levels.items(),
            key=lambda x: abs(x[1] - current_price)
        )
        
        # 현재 가격과 레벨 간의 거리 계산
        distance = abs(current_price - nearest_level[1]) / (high - low)
        
        return FibonacciLevels(
            levels=levels,
            high=high,
            low=low,
            is_uptrend=is_uptrend,
            current_price=current_price,
            nearest_level=nearest_level[0],
            distance_to_level=distance
        )

    def find_retracement_levels(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Tuple[float, float, bool]:
        """
        최근 고점/저점 자동 감지
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            window (int): 감지 윈도우 크기
            
        Returns:
            Tuple[float, float, bool]: (고점, 저점, 상승추세 여부)
        """
        # 최근 고점/저점 찾기
        recent_high = df['high'].rolling(window=window).max().iloc[-1]
        recent_low = df['low'].rolling(window=window).min().iloc[-1]
        
        # 추세 판단 (단순 이동평균선 기반)
        ma20 = df['close'].rolling(window=20).mean()
        ma50 = df['close'].rolling(window=50).mean()
        is_uptrend = ma20.iloc[-1] > ma50.iloc[-1]
        
        return recent_high, recent_low, is_uptrend

    def get_fibonacci_signals(
        self,
        df: pd.DataFrame,
        levels: FibonacciLevels,
        is_uptrend: bool
    ) -> FibonacciSignal:
        """
        레벨 기반 신호 생성
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            levels (FibonacciLevels): 피보나치 레벨 정보
            is_uptrend (bool): 상승 추세 여부
            
        Returns:
            FibonacciSignal: 생성된 신호
        """
        current_price = df['close'].iloc[-1]
        
        # 현재 가격이 레벨에 얼마나 가까운지 확인
        level_distances = {
            level: abs(price - current_price) / (levels.high - levels.low)
            for level, price in levels.levels.items()
        }
        
        # 가장 가까운 레벨 찾기
        nearest_level = min(level_distances.items(), key=lambda x: x[1])
        
        # 신호 강도 계산
        strength = 1 - min(nearest_level[1] / self.price_threshold, 1.0)
        
        # 신호 타입 결정
        if is_uptrend:
            # 상승 추세에서는 레벨이 지지선으로 작용
            if current_price <= levels.levels[nearest_level[0]]:
                signal_type = 'BUY'
                reason = f"가격이 {nearest_level[0].value} 피보나치 지지선 근처"
            else:
                signal_type = 'NEUTRAL'
                reason = "가격이 피보나치 레벨과 충분히 떨어져 있음"
        else:
            # 하락 추세에서는 레벨이 저항선으로 작용
            if current_price >= levels.levels[nearest_level[0]]:
                signal_type = 'SELL'
                reason = f"가격이 {nearest_level[0].value} 피보나치 저항선 근처"
            else:
                signal_type = 'NEUTRAL'
                reason = "가격이 피보나치 레벨과 충분히 떨어져 있음"
        
        return FibonacciSignal(
            type=signal_type,
            strength=strength,
            level=nearest_level[0],
            price=current_price,
            reason=reason
        )

    def plot_fibonacci_levels(
        self,
        df: pd.DataFrame,
        levels: FibonacciLevels,
        title: str = "Fibonacci Levels"
    ) -> None:
        """
        차트에 피보나치 레벨 표시
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            levels (FibonacciLevels): 피보나치 레벨 정보
            title (str): 차트 제목
        """
        plt.figure(figsize=(12, 6))
        
        # 가격 차트 그리기
        plt.plot(df.index, df['close'], label='Price', color='blue')
        
        # 피보나치 레벨 그리기
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        for (level, price), color in zip(levels.levels.items(), colors):
            plt.axhline(y=price, color=color, linestyle='--', 
                       label=f'Fib {level.value}')
            
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_level_weight(self, level: FibonacciLevel) -> float:
        """
        레벨별 가중치 반환
        
        Args:
            level (FibonacciLevel): 피보나치 레벨
            
        Returns:
            float: 레벨의 가중치
        """
        return self.level_weights.get(level, 0.1)  # 기본값 0.1

    def is_price_near_level(
        self,
        price: float,
        level_price: float,
        high: float,
        low: float
    ) -> bool:
        """
        가격이 레벨 근처에 있는지 확인
        
        Args:
            price (float): 현재 가격
            level_price (float): 레벨 가격
            high (float): 고점
            low (float): 저점
            
        Returns:
            bool: 가격이 레벨 근처에 있는지 여부
        """
        price_range = high - low
        distance = abs(price - level_price) / price_range
        return distance <= self.price_threshold 