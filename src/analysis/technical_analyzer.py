"""
기술적 분석 모듈
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..utils.database import DatabaseManager

class TechnicalAnalyzer:
    """기술적 분석 클래스"""
    
    def __init__(self, db: DatabaseManager):
        """
        초기화
        
        Args:
            db (DatabaseManager): 데이터베이스 관리자
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
        
    async def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        기술적 지표 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Dict[str, Any]: 기술적 지표
        """
        try:
            indicators = {}
            
            # 볼린저 밴드
            indicators['bollinger'] = self._calculate_bollinger_bands(data)
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(data)
            
            # 이동평균선
            indicators['ma'] = self._calculate_moving_averages(data)
            
            # MACD
            indicators['macd'] = self._calculate_macd(data)
            
            # ATR
            indicators['atr'] = self._calculate_atr(data)
            
            # 거래량 프로파일
            indicators['volume_profile'] = self._calculate_volume_profile(data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {str(e)}")
            return {}
            
    def _calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        볼린저 밴드 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            window (int): 이동평균 기간
            num_std (float): 표준편차 배수
            
        Returns:
            Dict[str, pd.Series]: 볼린저 밴드
        """
        try:
            # 중간 밴드 (20일 이동평균)
            middle_band = data['close'].rolling(window=window).mean()
            
            # 표준편차
            std = data['close'].rolling(window=window).std()
            
            # 상단/하단 밴드
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }
            
        except Exception as e:
            self.logger.error(f"볼린저 밴드 계산 실패: {str(e)}")
            return {}
            
    def _calculate_rsi(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        RSI 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            period (int): RSI 기간
            
        Returns:
            pd.Series: RSI 값
        """
        try:
            # 가격 변화
            delta = data['close'].diff()
            
            # 상승/하락 구분
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # RS 계산
            rs = gain / loss
            
            # RSI 계산
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"RSI 계산 실패: {str(e)}")
            return pd.Series()
            
    def _calculate_moving_averages(
        self,
        data: pd.DataFrame,
        periods: list = [5, 10, 20, 60]
    ) -> Dict[str, pd.Series]:
        """
        이동평균선 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            periods (list): 이동평균 기간 리스트
            
        Returns:
            Dict[str, pd.Series]: 이동평균선
        """
        try:
            ma = {}
            
            for period in periods:
                ma[f'ma{period}'] = data['close'].rolling(window=period).mean()
                
            return ma
            
        except Exception as e:
            self.logger.error(f"이동평균선 계산 실패: {str(e)}")
            return {}
            
    def _calculate_macd(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            fast_period (int): 단기 이동평균 기간
            slow_period (int): 장기 이동평균 기간
            signal_period (int): 시그널 기간
            
        Returns:
            Dict[str, pd.Series]: MACD 지표
        """
        try:
            # 단기/장기 이동평균
            fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
            
            # MACD 라인
            macd_line = fast_ema - slow_ema
            
            # 시그널 라인
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # MACD 히스토그램
            macd_histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': macd_histogram
            }
            
        except Exception as e:
            self.logger.error(f"MACD 계산 실패: {str(e)}")
            return {}
            
    def _calculate_atr(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        ATR 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            period (int): ATR 기간
            
        Returns:
            pd.Series: ATR 값
        """
        try:
            # True Range 계산
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            # ATR 계산
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR 계산 실패: {str(e)}")
            return pd.Series()
            
    def _calculate_volume_profile(
        self,
        data: pd.DataFrame,
        price_bins: int = 20
    ) -> Dict[str, Any]:
        """
        거래량 프로파일 계산
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            price_bins (int): 가격 구간 수
            
        Returns:
            Dict[str, Any]: 거래량 프로파일
        """
        try:
            # 가격 구간 설정
            price_min = data['low'].min()
            price_max = data['high'].max()
            price_range = price_max - price_min
            bin_size = price_range / price_bins
            
            # 가격 구간별 거래량 집계
            volume_profile = {}
            for i in range(price_bins):
                price_level = price_min + (i * bin_size)
                mask = (data['low'] >= price_level) & (data['low'] < price_level + bin_size)
                volume = data.loc[mask, 'volume'].sum()
                volume_profile[price_level] = volume
                
            return volume_profile
            
        except Exception as e:
            self.logger.error(f"거래량 프로파일 계산 실패: {str(e)}")
            return {}
            
    async def analyze_trend(self, data: pd.DataFrame) -> str:
        """
        추세 분석
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            str: 추세 방향 (up/down/sideways)
        """
        try:
            # 이동평균선 계산
            ma = self._calculate_moving_averages(data)
            
            if len(ma) < 2:
                return 'sideways'
                
            # 단기/장기 이동평균 비교
            short_ma = list(ma.values())[0]
            long_ma = list(ma.values())[-1]
            
            if short_ma.iloc[-1] > long_ma.iloc[-1]:
                return 'up'
            elif short_ma.iloc[-1] < long_ma.iloc[-1]:
                return 'down'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"추세 분석 실패: {str(e)}")
            return 'sideways'
            
    async def analyze_momentum(self, data: pd.DataFrame) -> str:
        """
        모멘텀 분석
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            str: 모멘텀 상태 (overbought/oversold/neutral)
        """
        try:
            # RSI 계산
            rsi = self._calculate_rsi(data)
            
            if rsi.empty:
                return 'neutral'
                
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 70:
                return 'overbought'
            elif current_rsi < 30:
                return 'oversold'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"모멘텀 분석 실패: {str(e)}")
            return 'neutral'
            
    async def analyze_volatility(self, data: pd.DataFrame) -> str:
        """
        변동성 분석
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            str: 변동성 상태 (high/medium/low)
        """
        try:
            # ATR 계산
            atr = self._calculate_atr(data)
            
            if atr.empty:
                return 'medium'
                
            # ATR의 이동평균
            atr_ma = atr.rolling(window=20).mean()
            
            current_atr = atr.iloc[-1]
            avg_atr = atr_ma.iloc[-1]
            
            if current_atr > avg_atr * 1.5:
                return 'high'
            elif current_atr < avg_atr * 0.5:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            self.logger.error(f"변동성 분석 실패: {str(e)}")
            return 'medium' 