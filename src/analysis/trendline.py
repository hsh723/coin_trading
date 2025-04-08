import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from ..utils.logger import setup_logger
from .trend import TrendType

@dataclass
class Point:
    x: int  # 인덱스
    y: float  # 가격
    type: str  # 'HIGH' or 'LOW'

@dataclass
class Trendline:
    start_point: Point
    end_point: Point
    type: str  # 'UPTREND' or 'DOWNTREND'
    slope: float
    intercept: float
    touches: List[Point]
    strength: float  # 0.0 ~ 1.0

@dataclass
class TrendlineSignal:
    type: str  # 'LONG', 'SHORT', 'NEUTRAL'
    strength: float  # 0.0 ~ 1.0
    price: float
    stop_loss: float
    take_profit: float
    reason: str

class TrendlineIndicator:
    def __init__(self):
        self.logger = setup_logger()
        self.stop_loss_pct = 0.02  # 2% 손절
        self.take_profit_pct = 0.04  # 4% 익절
        self.min_touch_distance = 5  # 최소 터치 간격
        self.touch_threshold = 0.001  # 터치 판단 임계값 (0.1%)

    def detect_swing_points(
        self,
        df: pd.DataFrame,
        window: int = 5
    ) -> Tuple[List[Point], List[Point]]:
        """
        스윙 고점/저점 감지
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            window (int): 감지 윈도우 크기
            
        Returns:
            Tuple[List[Point], List[Point]]: (고점 리스트, 저점 리스트)
        """
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # 고점 감지
            if all(df['high'].iloc[i] > df['high'].iloc[i-window:i]) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+1:i+window+1]):
                highs.append(Point(i, df['high'].iloc[i], 'HIGH'))
                
            # 저점 감지
            if all(df['low'].iloc[i] < df['low'].iloc[i-window:i]) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+1:i+window+1]):
                lows.append(Point(i, df['low'].iloc[i], 'LOW'))
                
        return highs, lows

    def find_trendlines(
        self,
        df: pd.DataFrame,
        num_points: int = 3,
        window: int = 20
    ) -> List[Trendline]:
        """
        추세선 자동 감지
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            num_points (int): 추세선 구성에 필요한 최소 포인트 수
            window (int): 감지 윈도우 크기
            
        Returns:
            List[Trendline]: 감지된 추세선 리스트
        """
        trendlines = []
        highs, lows = self.detect_swing_points(df, window)
        
        # 상승 추세선 찾기
        for i in range(len(lows) - num_points + 1):
            points = lows[i:i+num_points]
            if self._is_valid_trendline(points, 'UPTREND'):
                trendline = self._create_trendline(points, 'UPTREND')
                trendlines.append(trendline)
                
        # 하락 추세선 찾기
        for i in range(len(highs) - num_points + 1):
            points = highs[i:i+num_points]
            if self._is_valid_trendline(points, 'DOWNTREND'):
                trendline = self._create_trendline(points, 'DOWNTREND')
                trendlines.append(trendline)
                
        return trendlines

    def find_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Tuple[List[float], List[float]]:
        """
        지지/저항 레벨 찾기
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            window (int): 감지 윈도우 크기
            
        Returns:
            Tuple[List[float], List[float]]: (지지 레벨 리스트, 저항 레벨 리스트)
        """
        highs, lows = self.detect_swing_points(df, window)
        
        # 지지 레벨 (저점들의 클러스터)
        support_levels = self._cluster_points([p.y for p in lows])
        
        # 저항 레벨 (고점들의 클러스터)
        resistance_levels = self._cluster_points([p.y for p in highs])
        
        return support_levels, resistance_levels

    def calculate_trendline_touch(
        self,
        df: pd.DataFrame,
        trendline: Trendline
    ) -> List[Point]:
        """
        추세선 터치 감지
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            trendline (Trendline): 추세선 정보
            
        Returns:
            List[Point]: 터치 포인트 리스트
        """
        touches = []
        
        for i in range(len(df)):
            price = df['close'].iloc[i]
            expected_price = trendline.slope * i + trendline.intercept
            
            # 터치 판단
            if abs(price - expected_price) / price < self.touch_threshold:
                touches.append(Point(i, price, 'TOUCH'))
                
        return touches

    def get_trendline_signals(self, df: pd.DataFrame) -> TrendlineSignal:
        """
        추세선 기반 매매 신호 생성
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            TrendlineSignal: 매매 신호
        """
        # 추세선 감지
        trendlines = self.find_trendlines(df)
        current_price = df['close'].iloc[-1]
        
        # 가장 강한 추세선 선택
        strongest_trendline = max(trendlines, key=lambda x: x.strength) if trendlines else None
        
        if strongest_trendline:
            # 터치 포인트 계산
            touches = self.calculate_trendline_touch(df, strongest_trendline)
            
            # 3번째 터치 확인
            if len(touches) >= 3 and touches[-1].x == len(df) - 1:
                if strongest_trendline.type == 'UPTREND':
                    signal_type = 'LONG'
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                    reason = "상승 추세선 3번째 터치 감지"
                else:
                    signal_type = 'SHORT'
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
                    reason = "하락 추세선 3번째 터치 감지"
            else:
                signal_type = 'NEUTRAL'
                stop_loss = take_profit = current_price
                reason = "추세선 터치 부족"
        else:
            signal_type = 'NEUTRAL'
            stop_loss = take_profit = current_price
            reason = "추세선 미감지"
            
        # 신호 강도 계산
        strength = self._calculate_signal_strength(strongest_trendline, touches) if strongest_trendline else 0.0
        
        return TrendlineSignal(
            type=signal_type,
            strength=strength,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason
        )

    def plot_trendlines(self, df: pd.DataFrame, trendlines: List[Trendline]):
        """
        추세선 시각화
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            trendlines (List[Trendline]): 감지된 추세선 리스트
        """
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['close'], label='Price', color='gray', alpha=0.5)
        
        for trendline in trendlines:
            x = np.array([trendline.start_point.x, trendline.end_point.x])
            y = np.array([trendline.start_point.y, trendline.end_point.y])
            plt.plot(x, y, label=f"{trendline.type} Line", color='red' if trendline.type == 'UPTREND' else 'blue')
            
            # 터치 포인트 표시
            for touch in trendline.touches:
                plt.scatter(touch.x, touch.y, color='green', s=100)
                
        plt.title('Trendlines Detection')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _is_valid_trendline(self, points: List[Point], trend_type: str) -> bool:
        """유효한 추세선인지 확인"""
        if len(points) < 3:
            return False
            
        # 포인트 간 최소 거리 확인
        for i in range(len(points) - 1):
            if points[i+1].x - points[i].x < self.min_touch_distance:
                return False
                
        # 추세 방향 확인
        if trend_type == 'UPTREND':
            return all(points[i].y < points[i+1].y for i in range(len(points)-1))
        else:
            return all(points[i].y > points[i+1].y for i in range(len(points)-1))

    def _create_trendline(self, points: List[Point], trend_type: str) -> Trendline:
        """추세선 객체 생성"""
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        
        # 선형 회귀로 기울기와 절편 계산
        slope, intercept = np.polyfit(x, y, 1)
        
        return Trendline(
            start_point=points[0],
            end_point=points[-1],
            type=trend_type,
            slope=slope,
            intercept=intercept,
            touches=points,
            strength=self._calculate_trendline_strength(points)
        )

    def _cluster_points(self, points: List[float], threshold: float = 0.02) -> List[float]:
        """포인트 클러스터링"""
        if not points:
            return []
            
        clusters = []
        current_cluster = [points[0]]
        
        for point in points[1:]:
            if abs(point - np.mean(current_cluster)) / np.mean(current_cluster) < threshold:
                current_cluster.append(point)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [point]
                
        clusters.append(np.mean(current_cluster))
        return clusters

    def _calculate_trendline_strength(self, points: List[Point]) -> float:
        """추세선 강도 계산"""
        if len(points) < 3:
            return 0.0
            
        # 포인트 수에 따른 강도
        point_strength = min(len(points) / 5, 1.0)
        
        # 포인트 간 거리에 따른 강도
        distances = [points[i+1].x - points[i].x for i in range(len(points)-1)]
        distance_strength = min(np.mean(distances) / 20, 1.0)
        
        # 가격 변동성에 따른 강도
        price_changes = [abs(points[i+1].y - points[i].y) / points[i].y for i in range(len(points)-1)]
        volatility_strength = min(np.mean(price_changes) / 0.05, 1.0)
        
        return (point_strength + distance_strength + volatility_strength) / 3

    def _calculate_signal_strength(self, trendline: Trendline, touches: List[Point]) -> float:
        """신호 강도 계산"""
        if not trendline or not touches:
            return 0.0
            
        # 추세선 강도
        trendline_strength = trendline.strength
        
        # 터치 횟수에 따른 강도
        touch_strength = min(len(touches) / 3, 1.0)
        
        # 마지막 터치의 정확도
        last_touch = touches[-1]
        expected_price = trendline.slope * last_touch.x + trendline.intercept
        accuracy_strength = 1 - abs(last_touch.y - expected_price) / last_touch.y
        
        return (trendline_strength + touch_strength + accuracy_strength) / 3 