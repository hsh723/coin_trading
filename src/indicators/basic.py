"""
기본 기술적 지표 클래스
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from src.utils.logger import get_logger

class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
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
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD 계산"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
        
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        middle = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower 