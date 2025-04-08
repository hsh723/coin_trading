import pandas as pd
import numpy as np
from typing import Tuple, List

class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """단순 이동평균 계산"""
        return data.rolling(window=period).mean()
        
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """지수 이동평균 계산"""
        return data.ewm(span=period, adjust=False).mean()
        
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    @staticmethod
    def calculate_macd(data: pd.Series,
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD 계산"""
        fast_ema = TechnicalIndicators.calculate_ema(data, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, slow_period)
        macd = fast_ema - slow_ema
        signal = TechnicalIndicators.calculate_ema(macd, signal_period)
        return macd, signal
        
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series,
                                period: int = 20,
                                std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        middle = TechnicalIndicators.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
        
    @staticmethod
    def calculate_stochastic(high: pd.Series,
                           low: pd.Series,
                           close: pd.Series,
                           k_period: int = 14,
                           d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """스토캐스틱 계산"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
        
    @staticmethod
    def calculate_atr(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = 14) -> pd.Series:
        """ATR 계산"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    @staticmethod
    def calculate_adx(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = 14) -> pd.Series:
        """ADX 계산"""
        tr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        # +DM, -DM 계산
        high_diff = high.diff()
        low_diff = low.diff()
        
        pos_dm = ((high_diff > 0) & (high_diff > -low_diff)) * high_diff
        neg_dm = ((low_diff < 0) & (-low_diff > high_diff)) * -low_diff
        
        # +DI, -DI 계산
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / tr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / tr)
        
        # ADX 계산
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
        
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV 계산"""
        obv = pd.Series(0, index=close.index)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
                
        return obv 