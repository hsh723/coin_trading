import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from ..utils.logger import setup_logger

class TrendType(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    NEUTRAL = "NEUTRAL"

@dataclass
class TrendInfo:
    trend_type: TrendType
    strength: float  # 0.0 ~ 1.0
    ma_short: float
    ma_long: float
    ma_distance: float
    is_crossover: bool
    crossover_direction: Optional[str]  # 'UP' or 'DOWN'

@dataclass
class TrendSignal:
    type: str  # 'LONG', 'SHORT', 'NEUTRAL'
    strength: float  # 0.0 ~ 1.0
    price: float
    stop_loss: float
    take_profit: float
    reason: str

class TrendIndicator:
    def __init__(self):
        self.logger = setup_logger()
        self.neutral_threshold = 0.002  # MA 간격이 이 값보다 작으면 중립으로 판단
        self.stop_loss_pct = 0.02  # 2% 손절
        self.take_profit_pct = 0.04  # 4% 익절

    def detect_trend(
        self,
        df: pd.DataFrame,
        short_period: int = 20,
        long_period: int = 60
    ) -> TrendInfo:
        """
        이동평균선 기반 트렌드 감지
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            short_period (int): 단기 이동평균선 기간
            long_period (int): 장기 이동평균선 기간
            
        Returns:
            TrendInfo: 트렌드 정보
        """
        # 이동평균선 계산
        ma_short = df['close'].rolling(window=short_period).mean()
        ma_long = df['close'].rolling(window=long_period).mean()
        
        # 현재 값
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        # MA 간격 계산
        ma_distance = (current_ma_short - current_ma_long) / current_ma_long
        
        # 크로스오버 감지
        is_crossover = self.detect_crossover(df, short_period, long_period)
        crossover_direction = self._get_crossover_direction(df, short_period, long_period)
        
        # 트렌드 타입 결정
        if abs(ma_distance) < self.neutral_threshold:
            trend_type = TrendType.NEUTRAL
        elif current_ma_short > current_ma_long:
            trend_type = TrendType.UPTREND
        else:
            trend_type = TrendType.DOWNTREND
            
        # 트렌드 강도 계산
        strength = self.calculate_trend_strength(df)
        
        return TrendInfo(
            trend_type=trend_type,
            strength=strength,
            ma_short=current_ma_short,
            ma_long=current_ma_long,
            ma_distance=ma_distance,
            is_crossover=is_crossover,
            crossover_direction=crossover_direction
        )

    def detect_crossover(
        self,
        df: pd.DataFrame,
        short_period: int = 20,
        long_period: int = 60
    ) -> bool:
        """
        MA 크로스오버 감지
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            short_period (int): 단기 이동평균선 기간
            long_period (int): 장기 이동평균선 기간
            
        Returns:
            bool: 크로스오버 발생 여부
        """
        ma_short = df['close'].rolling(window=short_period).mean()
        ma_long = df['close'].rolling(window=long_period).mean()
        
        # 현재와 이전 봉의 MA 비교
        current_short = ma_short.iloc[-1]
        current_long = ma_long.iloc[-1]
        prev_short = ma_short.iloc[-2]
        prev_long = ma_long.iloc[-2]
        
        # 크로스오버 조건 확인
        crossover_up = current_short > current_long and prev_short <= prev_long
        crossover_down = current_short < current_long and prev_short >= prev_long
        
        return crossover_up or crossover_down

    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        트렌드 강도 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            float: 트렌드 강도 (0.0 ~ 1.0)
        """
        # ATR(Average True Range) 계산
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # 현재 ATR과 평균 ATR 비교
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        
        # 강도 계산 (0.0 ~ 1.0)
        strength = min(current_atr / avg_atr, 1.0)
        
        return strength

    def get_trend_signals(self, df: pd.DataFrame) -> TrendSignal:
        """
        트렌드 기반 매매 신호 생성
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            TrendSignal: 매매 신호
        """
        # 트렌드 감지
        trend_info = self.detect_trend(df)
        current_price = df['close'].iloc[-1]
        
        # 신호 타입 결정
        if trend_info.trend_type == TrendType.UPTREND:
            signal_type = 'LONG'
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            reason = "상승 추세 감지"
        elif trend_info.trend_type == TrendType.DOWNTREND:
            signal_type = 'SHORT'
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
            reason = "하락 추세 감지"
        else:
            signal_type = 'NEUTRAL'
            stop_loss = take_profit = current_price
            reason = "중립 추세"
            
        # 크로스오버 정보 추가
        if trend_info.is_crossover:
            reason += f" (MA 크로스오버 {trend_info.crossover_direction})"
            
        return TrendSignal(
            type=signal_type,
            strength=trend_info.strength,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason
        )

    def _get_crossover_direction(
        self,
        df: pd.DataFrame,
        short_period: int,
        long_period: int
    ) -> Optional[str]:
        """
        크로스오버 방향 확인
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            short_period (int): 단기 이동평균선 기간
            long_period (int): 장기 이동평균선 기간
            
        Returns:
            Optional[str]: 'UP' 또는 'DOWN' 또는 None
        """
        ma_short = df['close'].rolling(window=short_period).mean()
        ma_long = df['close'].rolling(window=long_period).mean()
        
        current_short = ma_short.iloc[-1]
        current_long = ma_long.iloc[-1]
        prev_short = ma_short.iloc[-2]
        prev_long = ma_long.iloc[-2]
        
        if current_short > current_long and prev_short <= prev_long:
            return 'UP'
        elif current_short < current_long and prev_short >= prev_long:
            return 'DOWN'
        else:
            return None 