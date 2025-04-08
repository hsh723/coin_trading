"""
데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from src.analysis.indicators.technical import TechnicalIndicators

class DataProcessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        """데이터 전처리기 초기화"""
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        self.indicators = TechnicalIndicators()
        
    def _setup_logging(self):
        """로깅 설정"""
        self.logger.setLevel(logging.INFO)
        
    def preprocess_data(
        self,
        df: pd.DataFrame,
        indicators: List[str] = None
    ) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            df (pd.DataFrame): 원본 데이터
            indicators (List[str]): 계산할 지표 목록
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        try:
            # 데이터 복사
            processed_df = df.copy()
            
            # 기본 지표 계산
            processed_df = self._calculate_basic_indicators(processed_df)
            
            # 기술적 지표 계산
            if indicators:
                processed_df = self._calculate_technical_indicators(processed_df, indicators)
                
            # 결측치 처리
            processed_df = self._handle_missing_values(processed_df)
            
            # 이상치 처리
            processed_df = self._handle_outliers(processed_df)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 실패: {str(e)}")
            raise
            
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기본 지표 계산
        
        Args:
            df (pd.DataFrame): 원본 데이터
            
        Returns:
            pd.DataFrame: 기본 지표가 추가된 데이터
        """
        try:
            # 가격 변화율
            df['returns'] = df['close'].pct_change()
            
            # 로그 수익률
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 변동성
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 거래량 가중 가격
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            return df
            
        except Exception as e:
            self.logger.error(f"기본 지표 계산 실패: {str(e)}")
            raise
            
    def _calculate_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[str]
    ) -> pd.DataFrame:
        """
        기술적 지표 계산
        
        Args:
            df (pd.DataFrame): 원본 데이터
            indicators (List[str]): 계산할 지표 목록
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터
        """
        try:
            for indicator in indicators:
                if indicator == 'sma':
                    df['sma_20'] = self.indicators.calculate_sma(df['close'], 20)
                    df['sma_50'] = self.indicators.calculate_sma(df['close'], 50)
                    df['sma_200'] = self.indicators.calculate_sma(df['close'], 200)
                    
                elif indicator == 'ema':
                    df['ema_12'] = self.indicators.calculate_ema(df['close'], 12)
                    df['ema_26'] = self.indicators.calculate_ema(df['close'], 26)
                    
                elif indicator == 'rsi':
                    df['rsi'] = self.indicators.calculate_rsi(df['close'])
                    
                elif indicator == 'macd':
                    macd, signal = self.indicators.calculate_macd(df['close'])
                    df['macd'] = macd
                    df['macd_signal'] = signal
                    
                elif indicator == 'bollinger':
                    upper, middle, lower = self.indicators.calculate_bollinger_bands(df['close'])
                    df['bb_upper'] = upper
                    df['bb_middle'] = middle
                    df['bb_lower'] = lower
                    
            return df
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {str(e)}")
            raise
            
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치 처리
        
        Args:
            df (pd.DataFrame): 원본 데이터
            
        Returns:
            pd.DataFrame: 결측치가 처리된 데이터
        """
        try:
            # 전진 채우기
            df = df.fillna(method='ffill')
            
            # 남은 결측치는 0으로 채우기
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"결측치 처리 실패: {str(e)}")
            raise
            
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 처리
        
        Args:
            df (pd.DataFrame): 원본 데이터
            
        Returns:
            pd.DataFrame: 이상치가 처리된 데이터
        """
        try:
            # Z-score 기반 이상치 제거
            for column in df.columns:
                if column not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                    df.loc[z_scores > 3, column] = df[column].mean()
                    
            return df
            
        except Exception as e:
            self.logger.error(f"이상치 처리 실패: {str(e)}")
            raise

# 전역 인스턴스 생성
data_processor = DataProcessor() 