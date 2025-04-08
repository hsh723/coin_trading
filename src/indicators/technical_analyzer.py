"""
기술적 분석기 클래스
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.analysis.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

class TechnicalAnalysisError(Exception):
    """기술적 분석 관련 기본 예외"""
    pass

class DataNotFoundError(TechnicalAnalysisError):
    """데이터를 찾을 수 없을 때 발생하는 예외"""
    pass

class CalculationError(TechnicalAnalysisError):
    """계산 중 오류가 발생할 때 발생하는 예외"""
    pass

class TechnicalAnalyzer:
    """기술적 분석기 클래스"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        기술적 지표 분석
        
        Args:
            data (pd.DataFrame): 가격 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
            
        Raises:
            DataNotFoundError: 데이터가 비어있을 때
            CalculationError: 계산 중 오류가 발생할 때
        """
        if data.empty:
            raise DataNotFoundError("분석할 데이터가 없습니다.")
            
        try:
            # RSI 계산
            rsi = TechnicalIndicators.calculate_rsi(data['close'], 14)
            
            # MACD 계산
            macd, signal = TechnicalIndicators.calculate_macd(
                data['close'],
                12,  # fast
                26,  # slow
                9    # signal
            )
            
            # 볼린저 밴드 계산
            upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
                data['close'],
                20,  # period
                2    # std
            )
            
            # 이동평균 계산
            sma_20 = TechnicalIndicators.calculate_sma(data['close'], 20)
            ema_20 = TechnicalIndicators.calculate_ema(data['close'], 20)
            
            return {
                'rsi': rsi,
                'macd': macd,
                'signal': signal,
                'bollinger_upper': upper,
                'bollinger_middle': middle,
                'bollinger_lower': lower,
                'sma_20': sma_20,
                'ema_20': ema_20
            }
            
        except Exception as e:
            self.logger.error(f"기술적 분석 중 오류 발생: {str(e)}")
            raise CalculationError(f"기술적 분석 중 오류 발생: {str(e)}")
            
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        매매 신호 생성
        
        Args:
            data (pd.DataFrame): 가격 데이터
            
        Returns:
            Dict[str, Any]: 매매 신호
            
        Raises:
            DataNotFoundError: 데이터가 비어있을 때
            CalculationError: 계산 중 오류가 발생할 때
        """
        if data.empty:
            raise DataNotFoundError("신호를 생성할 데이터가 없습니다.")
            
        try:
            analysis = self.analyze(data)
            latest = data.iloc[-1]
            
            # RSI 기반 신호
            rsi_signal = 'neutral'
            if analysis['rsi'].iloc[-1] > 70:
                rsi_signal = 'overbought'
            elif analysis['rsi'].iloc[-1] < 30:
                rsi_signal = 'oversold'
                
            # MACD 기반 신호
            macd_signal = 'neutral'
            if analysis['macd'].iloc[-1] > analysis['signal'].iloc[-1]:
                macd_signal = 'bullish'
            elif analysis['macd'].iloc[-1] < analysis['signal'].iloc[-1]:
                macd_signal = 'bearish'
                
            # 볼린저 밴드 기반 신호
            bb_signal = 'neutral'
            if latest['close'] > analysis['bollinger_upper'].iloc[-1]:
                bb_signal = 'overbought'
            elif latest['close'] < analysis['bollinger_lower'].iloc[-1]:
                bb_signal = 'oversold'
                
            return {
                'rsi': rsi_signal,
                'macd': macd_signal,
                'bollinger': bb_signal,
                'price': latest['close']
            }
            
        except Exception as e:
            self.logger.error(f"신호 생성 중 오류 발생: {str(e)}")
            raise CalculationError(f"신호 생성 중 오류 발생: {str(e)}") 