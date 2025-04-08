import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
from ..utils.logger import setup_logger
from .trend import TrendType

@dataclass
class StochasticInfo:
    k: float  # %K 값
    d: float  # %D 값
    is_oversold: bool
    is_overbought: bool
    crossover_up: bool
    crossover_down: bool
    trend_type: TrendType

@dataclass
class StochasticSignal:
    type: str  # 'LONG', 'SHORT', 'NEUTRAL'
    strength: float  # 0.0 ~ 1.0
    price: float
    stop_loss: float
    take_profit: float
    reason: str

class StochasticIndicator:
    def __init__(self):
        self.logger = setup_logger()
        self.stop_loss_pct = 0.02  # 2% 손절
        self.take_profit_pct = 0.04  # 4% 익절
        self.oversold_threshold = 20
        self.overbought_threshold = 80
        self.crossover_threshold = 0.5  # 크로스오버 판단 임계값

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        slowing: int = 3
    ) -> StochasticInfo:
        """
        스토캐스틱 오실레이터 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            k_period (int): %K 기간
            d_period (int): %D 기간
            slowing (int): 슬로잉 기간
            
        Returns:
            StochasticInfo: 스토캐스틱 정보
        """
        # 최고가/최저가 계산
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # %K 계산
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # 슬로잉 적용
        k = k.rolling(window=slowing).mean()
        
        # %D 계산
        d = k.rolling(window=d_period).mean()
        
        # 현재 값
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        
        # 과매수/과매도 확인
        is_oversold = self.is_oversold(current_k)
        is_overbought = self.is_overbought(current_k)
        
        # 크로스오버 확인
        crossover_up = self._detect_crossover_up(k, d)
        crossover_down = self._detect_crossover_down(k, d)
        
        # 추세 정보 가져오기
        trend_type = self._get_trend_type(df)
        
        return StochasticInfo(
            k=current_k,
            d=current_d,
            is_oversold=is_oversold,
            is_overbought=is_overbought,
            crossover_up=crossover_up,
            crossover_down=crossover_down,
            trend_type=trend_type
        )

    def is_oversold(self, value: float, threshold: int = 20) -> bool:
        """
        과매도 상태 확인
        
        Args:
            value (float): 스토캐스틱 값
            threshold (int): 과매도 임계값
            
        Returns:
            bool: 과매도 여부
        """
        return value <= threshold

    def is_overbought(self, value: float, threshold: int = 80) -> bool:
        """
        과매수 상태 확인
        
        Args:
            value (float): 스토캐스틱 값
            threshold (int): 과매수 임계값
            
        Returns:
            bool: 과매수 여부
        """
        return value >= threshold

    def get_stochastic_signals(self, df: pd.DataFrame) -> StochasticSignal:
        """
        스토캐스틱 기반 매매 신호 생성
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            StochasticSignal: 매매 신호
        """
        # 스토캐스틱 계산
        stoch_info = self.calculate_stochastic(df)
        current_price = df['close'].iloc[-1]
        
        # 신호 타입 결정
        if stoch_info.trend_type == TrendType.UPTREND and stoch_info.is_oversold:
            signal_type = 'LONG'
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            reason = "상승 추세에서 과매도 상태 감지"
        elif stoch_info.trend_type == TrendType.DOWNTREND and stoch_info.is_overbought:
            signal_type = 'SHORT'
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
            reason = "하락 추세에서 과매수 상태 감지"
        else:
            signal_type = 'NEUTRAL'
            stop_loss = take_profit = current_price
            reason = "신호 없음"
            
        # 크로스오버 정보 추가
        if stoch_info.crossover_up:
            reason += " (스토캐스틱 상향 크로스오버)"
        elif stoch_info.crossover_down:
            reason += " (스토캐스틱 하향 크로스오버)"
            
        # 신호 강도 계산
        strength = self._calculate_signal_strength(stoch_info)
        
        return StochasticSignal(
            type=signal_type,
            strength=strength,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason
        )

    def _detect_crossover_up(self, k: pd.Series, d: pd.Series) -> bool:
        """상향 크로스오버 감지"""
        return (
            k.iloc[-1] > d.iloc[-1] and
            k.iloc[-2] <= d.iloc[-2] and
            abs(k.iloc[-1] - d.iloc[-1]) > self.crossover_threshold
        )

    def _detect_crossover_down(self, k: pd.Series, d: pd.Series) -> bool:
        """하향 크로스오버 감지"""
        return (
            k.iloc[-1] < d.iloc[-1] and
            k.iloc[-2] >= d.iloc[-2] and
            abs(k.iloc[-1] - d.iloc[-1]) > self.crossover_threshold
        )

    def _get_trend_type(self, df: pd.DataFrame) -> TrendType:
        """추세 타입 확인"""
        ma20 = df['close'].rolling(window=20).mean()
        ma60 = df['close'].rolling(window=60).mean()
        
        if ma20.iloc[-1] > ma60.iloc[-1]:
            return TrendType.UPTREND
        elif ma20.iloc[-1] < ma60.iloc[-1]:
            return TrendType.DOWNTREND
        else:
            return TrendType.NEUTRAL

    def _calculate_signal_strength(self, stoch_info: StochasticInfo) -> float:
        """신호 강도 계산"""
        strength = 0.0
        
        # 과매수/과매도 상태에 따른 강도
        if stoch_info.is_oversold:
            strength += 0.4
        elif stoch_info.is_overbought:
            strength += 0.4
            
        # 크로스오버에 따른 강도
        if stoch_info.crossover_up or stoch_info.crossover_down:
            strength += 0.3
            
        # 추세와의 일치성에 따른 강도
        if (stoch_info.trend_type == TrendType.UPTREND and stoch_info.is_oversold) or \
           (stoch_info.trend_type == TrendType.DOWNTREND and stoch_info.is_overbought):
            strength += 0.3
            
        return min(strength, 1.0) 