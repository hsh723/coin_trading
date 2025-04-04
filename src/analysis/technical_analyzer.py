"""
기술적 분석 모듈
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Any, Optional
import logging
from ..utils.database import DatabaseManager

class TechnicalAnalyzer:
    """기술적 분석 클래스"""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """초기화"""
        self.db = db
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        기술적 지표 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Dict[str, Any]: 기술적 지표
        """
        try:
            # 기본 지표 계산
            indicators = {
                'bb': self._calculate_bollinger_bands(df),
                'rsi': self._calculate_rsi(df),
                'macd': self._calculate_macd(df),
                'atr': self._calculate_atr(df),
                'volume_profile': self._calculate_volume_profile(df)
            }
            
            # 추가 분석
            indicators.update({
                'trend': self._analyze_trend(df, indicators),
                'momentum': self._analyze_momentum(df, indicators),
                'volatility': self._analyze_volatility(df, indicators)
            })
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {str(e)}")
            return {}
            
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산"""
        try:
            # 20일 이동평균
            sma = ta.volatility.bollinger_mavg(df['close'], window=20)
            
            # 표준편차
            std = ta.volatility.bollinger_hband(df['close'], window=20) - sma
            
            # 상단/하단 밴드
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            return {
                'sma': sma,
                'upper': upper_band,
                'lower': lower_band
            }
            
        except Exception as e:
            self.logger.error(f"볼린저 밴드 계산 실패: {str(e)}")
            return {}
            
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """RSI 계산"""
        try:
            return ta.momentum.rsi(df['close'], window=14)
        except Exception as e:
            self.logger.error(f"RSI 계산 실패: {str(e)}")
            return pd.Series()
            
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """MACD 계산"""
        try:
            macd = ta.trend.MACD(df['close'])
            return {
                'macd': macd.macd(),
                'signal': macd.macd_signal(),
                'histogram': macd.macd_diff()
            }
        except Exception as e:
            self.logger.error(f"MACD 계산 실패: {str(e)}")
            return {}
            
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR 계산"""
        try:
            return ta.volatility.average_true_range(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
        except Exception as e:
            self.logger.error(f"ATR 계산 실패: {str(e)}")
            return pd.Series()
            
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """거래량 프로파일 계산"""
        try:
            # 가격 구간별 거래량 집계
            price_bins = pd.cut(df['close'], bins=20)
            volume_profile = df.groupby(price_bins)['volume'].sum()
            
            return {
                'profile': volume_profile,
                'poc': volume_profile.idxmax(),  # Point of Control
                'value_area': self._calculate_value_area(volume_profile)
            }
        except Exception as e:
            self.logger.error(f"거래량 프로파일 계산 실패: {str(e)}")
            return {}
            
    def _calculate_value_area(self, volume_profile: pd.Series) -> Dict[str, float]:
        """Value Area 계산"""
        try:
            total_volume = volume_profile.sum()
            target_volume = total_volume * 0.68  # 68% 규칙
            
            sorted_profile = volume_profile.sort_values(ascending=False)
            cumulative_volume = sorted_profile.cumsum()
            
            value_area = sorted_profile[cumulative_volume <= target_volume]
            
            return {
                'high': value_area.index.max(),
                'low': value_area.index.min()
            }
        except Exception as e:
            self.logger.error(f"Value Area 계산 실패: {str(e)}")
            return {}
            
    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """추세 분석"""
        try:
            # 이동평균선 기반 추세
            sma_20 = indicators['bb']['sma']
            sma_50 = ta.trend.sma_indicator(df['close'], window=50)
            sma_200 = ta.trend.sma_indicator(df['close'], window=200)
            
            # 추세 강도
            trend_strength = self._calculate_trend_strength(df, sma_20)
            
            return {
                'direction': 'up' if sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1] else 'down',
                'strength': trend_strength,
                'sma_cross': self._check_sma_cross(sma_20, sma_50)
            }
        except Exception as e:
            self.logger.error(f"추세 분석 실패: {str(e)}")
            return {}
            
    def _calculate_trend_strength(self, df: pd.DataFrame, sma: pd.Series) -> float:
        """추세 강도 계산"""
        try:
            # ADX 지표 사용
            adx = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            return adx.adx().iloc[-1] / 100.0  # 0~1 사이 값으로 정규화
        except Exception as e:
            self.logger.error(f"추세 강도 계산 실패: {str(e)}")
            return 0.0
            
    def _check_sma_cross(self, sma_short: pd.Series, sma_long: pd.Series) -> str:
        """이동평균선 교차 확인"""
        try:
            if sma_short.iloc[-2] <= sma_long.iloc[-2] and sma_short.iloc[-1] > sma_long.iloc[-1]:
                return 'golden_cross'
            elif sma_short.iloc[-2] >= sma_long.iloc[-2] and sma_short.iloc[-1] < sma_long.iloc[-1]:
                return 'death_cross'
            return 'none'
        except Exception as e:
            self.logger.error(f"이동평균선 교차 확인 실패: {str(e)}")
            return 'none'
            
    def _analyze_momentum(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """모멘텀 분석"""
        try:
            rsi = indicators['rsi']
            macd = indicators['macd']
            
            return {
                'rsi_signal': 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral',
                'macd_signal': 'bullish' if macd['histogram'].iloc[-1] > 0 else 'bearish',
                'strength': abs(macd['histogram'].iloc[-1]) / macd['histogram'].std()
            }
        except Exception as e:
            self.logger.error(f"모멘텀 분석 실패: {str(e)}")
            return {}
            
    def _analyze_volatility(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """변동성 분석"""
        try:
            atr = indicators['atr']
            bb = indicators['bb']
            
            # 변동성 지수
            volatility_index = atr.iloc[-1] / df['close'].iloc[-1]
            
            # 볼린저 밴드 수축/확장
            bb_width = (bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) / bb['sma'].iloc[-1]
            bb_width_ma = bb_width.rolling(window=20).mean().iloc[-1]
            
            return {
                'index': volatility_index,
                'bb_squeeze': bb_width < bb_width_ma * 0.5,
                'bb_expansion': bb_width > bb_width_ma * 1.5
            }
        except Exception as e:
            self.logger.error(f"변동성 분석 실패: {str(e)}")
            return {} 