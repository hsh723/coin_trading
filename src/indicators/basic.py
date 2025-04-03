"""
기본 기술적 지표 모듈
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict
from src.utils.logger import setup_logger

logger = setup_logger()

class TechnicalIndicators:
    """
    기본 기술적 지표 계산 클래스
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        단순 이동평균 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            
        Returns:
            pd.Series: 이동평균
        """
        try:
            return data.rolling(window=period).mean()
        except Exception as e:
            logger.error(f"SMA 계산 실패: {str(e)}")
            return pd.Series(index=data.index)
            
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        지수 이동평균 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            
        Returns:
            pd.Series: 지수 이동평균
        """
        try:
            return data.ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"EMA 계산 실패: {str(e)}")
            return pd.Series(index=data.index)
            
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI(상대강도지수) 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            
        Returns:
            pd.Series: RSI
        """
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"RSI 계산 실패: {str(e)}")
            return pd.Series(index=data.index)
            
    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD(이동평균수렴확산) 계산
        
        Args:
            data (pd.Series): 가격 데이터
            fast_period (int): 단기 기간
            slow_period (int): 장기 기간
            signal_period (int): 시그널 기간
            
        Returns:
            Dict[str, pd.Series]: MACD 라인, 시그널 라인, 히스토그램
        """
        try:
            fast_ema = TechnicalIndicators.ema(data, fast_period)
            slow_ema = TechnicalIndicators.ema(data, slow_period)
            
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"MACD 계산 실패: {str(e)}")
            return {
                'macd': pd.Series(index=data.index),
                'signal': pd.Series(index=data.index),
                'histogram': pd.Series(index=data.index)
            }
            
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        볼린저 밴드 계산
        
        Args:
            data (pd.Series): 가격 데이터
            period (int): 기간
            std_dev (float): 표준편차 승수
            
        Returns:
            Dict[str, pd.Series]: 상단, 중간, 하단 밴드
        """
        try:
            middle = TechnicalIndicators.sma(data, period)
            std = data.rolling(window=period).std()
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 실패: {str(e)}")
            return {
                'upper': pd.Series(index=data.index),
                'middle': pd.Series(index=data.index),
                'lower': pd.Series(index=data.index)
            }
            
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        스토캐스틱 오실레이터 계산
        
        Args:
            high (pd.Series): 고가
            low (pd.Series): 저가
            close (pd.Series): 종가
            k_period (int): %K 기간
            d_period (int): %D 기간
            
        Returns:
            Dict[str, pd.Series]: %K, %D
        """
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d = k.rolling(window=d_period).mean()
            
            return {
                'k': k,
                'd': d
            }
        except Exception as e:
            logger.error(f"스토캐스틱 계산 실패: {str(e)}")
            return {
                'k': pd.Series(index=close.index),
                'd': pd.Series(index=close.index)
            }
            
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        ATR(평균진폭) 계산
        
        Args:
            high (pd.Series): 고가
            low (pd.Series): 저가
            close (pd.Series): 종가
            period (int): 기간
            
        Returns:
            pd.Series: ATR
        """
        try:
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"ATR 계산 실패: {str(e)}")
            return pd.Series(index=close.index)
            
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        ADX(평균방향지수) 계산
        
        Args:
            high (pd.Series): 고가
            low (pd.Series): 저가
            close (pd.Series): 종가
            period (int): 기간
            
        Returns:
            Dict[str, pd.Series]: ADX, +DI, -DI
        """
        try:
            tr = pd.DataFrame()
            tr['h-l'] = high - low
            tr['h-pc'] = abs(high - close.shift(1))
            tr['l-pc'] = abs(low - close.shift(1))
            tr['tr'] = tr.max(axis=1)
            
            up = high - high.shift(1)
            down = low.shift(1) - low
            
            pos_dm = pd.Series(0.0, index=up.index, dtype=float)
            neg_dm = pd.Series(0.0, index=down.index, dtype=float)
            
            pos_dm[up > down] = up[up > down]
            neg_dm[down > up] = down[down > up]
            
            tr14 = tr['tr'].rolling(window=period).mean()
            pos_di14 = 100 * (pos_dm.rolling(window=period).mean() / tr14)
            neg_di14 = 100 * (neg_dm.rolling(window=period).mean() / tr14)
            
            dx = 100 * abs(pos_di14 - neg_di14) / (pos_di14 + neg_di14)
            adx = dx.rolling(window=period).mean()
            
            return {
                'adx': adx,
                'pos_di': pos_di14,
                'neg_di': neg_di14
            }
        except Exception as e:
            logger.error(f"ADX 계산 실패: {str(e)}")
            return {
                'adx': pd.Series(index=close.index),
                'pos_di': pd.Series(index=close.index),
                'neg_di': pd.Series(index=close.index)
            }
            
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        OBV(거래량가중이동평균) 계산
        
        Args:
            close (pd.Series): 종가
            volume (pd.Series): 거래량
            
        Returns:
            pd.Series: OBV
        """
        try:
            price_change = close.diff()
            obv = pd.Series(0, index=close.index)
            
            obv[price_change > 0] = volume[price_change > 0]
            obv[price_change < 0] = -volume[price_change < 0]
            
            return obv.cumsum()
        except Exception as e:
            logger.error(f"OBV 계산 실패: {str(e)}")
            return pd.Series(index=close.index)
            
    @staticmethod
    def ichimoku(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52
    ) -> Dict[str, pd.Series]:
        """
        일목균형표 (Ichimoku Cloud)
        
        Args:
            high (pd.Series): 고가 데이터
            low (pd.Series): 저가 데이터
            close (pd.Series): 종가 데이터
            tenkan_period (int): 전환선 기간
            kijun_period (int): 기준선 기간
            senkou_b_period (int): 선행스팬 B 기간
            
        Returns:
            Dict[str, pd.Series]: 일목균형표 값 (tenkan, kijun, senkou_a, senkou_b, chikou)
        """
        try:
            # 전환선 (Tenkan-sen)
            tenkan_high = high.rolling(window=tenkan_period).max()
            tenkan_low = low.rolling(window=tenkan_period).min()
            tenkan = (tenkan_high + tenkan_low) / 2
            
            # 기준선 (Kijun-sen)
            kijun_high = high.rolling(window=kijun_period).max()
            kijun_low = low.rolling(window=kijun_period).min()
            kijun = (kijun_high + kijun_low) / 2
            
            # 선행스팬 A (Senkou Span A)
            senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
            
            # 선행스팬 B (Senkou Span B)
            senkou_b_high = high.rolling(window=senkou_b_period).max()
            senkou_b_low = low.rolling(window=senkou_b_period).min()
            senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
            
            # 후행스팬 (Chikou Span)
            chikou = close.shift(-kijun_period)
            
            return {
                'tenkan': tenkan,
                'kijun': kijun,
                'senkou_a': senkou_a,
                'senkou_b': senkou_b,
                'chikou': chikou
            }
        except Exception as e:
            logger.error(f"일목균형표 계산 실패: {str(e)}")
            return {
                'tenkan': pd.Series(index=close.index),
                'kijun': pd.Series(index=close.index),
                'senkou_a': pd.Series(index=close.index),
                'senkou_b': pd.Series(index=close.index),
                'chikou': pd.Series(index=close.index)
            } 