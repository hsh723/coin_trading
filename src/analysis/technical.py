import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import talib

class TechnicalAnalyzer:
    def __init__(self):
        self.data = None
    
    def set_data(self, data: pd.DataFrame) -> None:
        """데이터 설정"""
        self.data = data

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """RSI(Relative Strength Index) 계산"""
        return talib.RSI(self.data['close'], timeperiod=period)

    def calculate_macd(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD(Moving Average Convergence Divergence) 계산"""
        macd, signal, hist = talib.MACD(
            self.data['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        return macd, signal, hist

    def calculate_bollinger_bands(self, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        upper, middle, lower = talib.BBANDS(
            self.data['close'],
            timeperiod=period
        )
        return upper, middle, lower

    def identify_trend(self, period: int = 20) -> str:
        """추세 식별"""
        sma = talib.SMA(self.data['close'], timeperiod=period)
        current_price = self.data['close'].iloc[-1]
        if current_price > sma.iloc[-1]:
            return "UPTREND"
        else:
            return "DOWNTREND"

    def find_support_resistance(self) -> Dict[str, float]:
        """지지/저항 레벨 감지"""
        price_series = self.data['close']
        window = 20
        
        highs = price_series.rolling(window=window, center=True).max()
        lows = price_series.rolling(window=window, center=True).min()
        
        resistance_levels = self._find_levels(highs)
        support_levels = self._find_levels(lows)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def _find_levels(self, price_series: pd.Series) -> List[float]:
        """주요 가격 레벨 식별"""
        levels = []
        tolerance = 0.02  # 2% 허용 오차
        
        for i in range(1, len(price_series) - 1):
            if abs(price_series[i] - price_series[i-1]) <= tolerance * price_series[i]:
                levels.append(price_series[i])
                
        return sorted(list(set([round(level, 2) for level in levels])))

    def calculate_ichimoku(self) -> Dict[str, pd.Series]:
        """일목균형표 계산"""
        high = self.data['high']
        low = self.data['low']
        
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b
        }

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """Average True Range 계산"""
        return talib.ATR(self.data['high'], self.data['low'], self.data['close'], timeperiod=period)

    def identify_candlestick_patterns(self) -> Dict[str, int]:
        """캔들스틱 패턴 식별"""
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'engulfing': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR
        }
        
        results = {}
        for name, pattern_func in patterns.items():
            results[name] = pattern_func(
                self.data['open'],
                self.data['high'],
                self.data['low'],
                self.data['close']
            )[-1]  # 최신 패턴만 반환
            
        return results
