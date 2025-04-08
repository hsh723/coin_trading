"""
데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from src.analysis.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

class DataProcessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        """데이터 전처리기 초기화"""
        self.logger = get_logger(__name__)
        self.indicators = TechnicalIndicators()
        
    def preprocess_data(
        self,
        df: pd.DataFrame,
        fill_missing: bool = True,
        remove_outliers: bool = False
    ) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            df (pd.DataFrame): 원본 데이터
            fill_missing (bool): 결측치 처리 여부
            remove_outliers (bool): 이상치 제거 여부
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        try:
            # 데이터 복사
            processed_df = df.copy()
            
            # 결측치 처리
            if fill_missing:
                processed_df = self._handle_missing_values(processed_df)
                
            # 이상치 처리
            if remove_outliers:
                processed_df = self._handle_outliers(processed_df)
                
            # 데이터 정규화
            processed_df = self._normalize_data(processed_df)
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 실패: {str(e)}")
            raise
            
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[str] = ['rsi', 'macd', 'bollinger']
    ) -> pd.DataFrame:
        """
        기술적 지표 추가
        
        Args:
            df (pd.DataFrame): 원본 데이터
            indicators (List[str]): 추가할 지표 목록
            
        Returns:
            pd.DataFrame: 지표가 추가된 데이터
        """
        try:
            # 데이터 복사
            df_with_indicators = df.copy()
            
            # 각 지표 계산
            for indicator in indicators:
                if indicator == 'rsi':
                    df_with_indicators['rsi'] = self.indicators.calculate_rsi(df['close'])
                    
                elif indicator == 'macd':
                    macd, signal = self.indicators.calculate_macd(df['close'])
                    df_with_indicators['macd'] = macd
                    df_with_indicators['macd_signal'] = signal
                    
                elif indicator == 'bollinger':
                    upper, middle, lower = self.indicators.calculate_bollinger_bands(df['close'])
                    df_with_indicators['bb_upper'] = upper
                    df_with_indicators['bb_middle'] = middle
                    df_with_indicators['bb_lower'] = lower
                    
                elif indicator == 'sma':
                    df_with_indicators['sma_20'] = self.indicators.calculate_sma(df['close'], 20)
                    df_with_indicators['sma_50'] = self.indicators.calculate_sma(df['close'], 50)
                    df_with_indicators['sma_200'] = self.indicators.calculate_sma(df['close'], 200)
                    
                elif indicator == 'ema':
                    df_with_indicators['ema_12'] = self.indicators.calculate_ema(df['close'], 12)
                    df_with_indicators['ema_26'] = self.indicators.calculate_ema(df['close'], 26)
                    
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"기술적 지표 추가 실패: {str(e)}")
            raise
            
    def resample_timeframe(
        self,
        df: pd.DataFrame,
        source_tf: str,
        target_tf: str
    ) -> pd.DataFrame:
        """
        타임프레임 리샘플링
        
        Args:
            df (pd.DataFrame): 원본 데이터
            source_tf (str): 원본 타임프레임
            target_tf (str): 목표 타임프레임
            
        Returns:
            pd.DataFrame: 리샘플링된 데이터
        """
        try:
            # 타임스탬프를 인덱스로 설정
            df = df.set_index('timestamp')
            
            # 리샘플링 규칙 설정
            rule = self._get_resample_rule(target_tf)
            
            # OHLCV 데이터 리샘플링
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # 인덱스 재설정
            resampled = resampled.reset_index()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"타임프레임 리샘플링 실패: {str(e)}")
            raise
            
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        학습용 데이터셋 생성
        
        Args:
            df (pd.DataFrame): 원본 데이터
            train_ratio (float): 훈련 데이터 비율
            val_ratio (float): 검증 데이터 비율
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 훈련/검증/테스트 데이터셋
        """
        try:
            # 데이터 정렬
            df = df.sort_values('timestamp')
            
            # 데이터 분할
            train_size = int(len(df) * train_ratio)
            val_size = int(len(df) * val_ratio)
            
            train_data = df[:train_size]
            val_data = df[train_size:train_size + val_size]
            test_data = df[train_size + val_size:]
            
            return train_data, val_data, test_data
            
        except Exception as e:
            self.logger.error(f"데이터셋 생성 실패: {str(e)}")
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
            
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정규화
        
        Args:
            df (pd.DataFrame): 원본 데이터
            
        Returns:
            pd.DataFrame: 정규화된 데이터
        """
        try:
            # 정규화할 컬럼 선택
            columns_to_normalize = ['open', 'high', 'low', 'close', 'volume']
            
            # Min-Max 정규화
            for column in columns_to_normalize:
                if column in df.columns:
                    min_val = df[column].min()
                    max_val = df[column].max()
                    df[column] = (df[column] - min_val) / (max_val - min_val)
                    
            return df
            
        except Exception as e:
            self.logger.error(f"데이터 정규화 실패: {str(e)}")
            raise
            
    def _get_resample_rule(self, timeframe: str) -> str:
        """
        리샘플링 규칙 생성
        
        Args:
            timeframe (str): 타임프레임
            
        Returns:
            str: 리샘플링 규칙
        """
        timeframe_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        return timeframe_map.get(timeframe, '1H')

# 전역 인스턴스 생성
data_processor = DataProcessor() 