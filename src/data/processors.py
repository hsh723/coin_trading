"""
데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime, timedelta
from ..utils.logger import setup_logger

class DataProcessor:
    """
    데이터 전처리를 수행하는 클래스
    """
    
    def __init__(self):
        """초기화"""
        self.logger = setup_logger()
        self.logger.info("DataProcessor initialized")
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = 'ffill',
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        결측값 처리
        
        Args:
            data (pd.DataFrame): 입력 데이터
            method (str): 결측값 처리 방법 ('ffill', 'bfill', 'interpolate')
            limit (int, optional): 연속 결측값 처리 제한
            
        Returns:
            pd.DataFrame: 결측값이 처리된 데이터
        """
        try:
            self.logger.info(f"Handling missing values using {method} method")
            
            # 결측값 개수 기록
            missing_before = data.isnull().sum()
            
            # 결측값 처리
            if method == 'ffill':
                data = data.fillna(method='ffill', limit=limit)
            elif method == 'bfill':
                data = data.fillna(method='bfill', limit=limit)
            elif method == 'interpolate':
                data = data.interpolate(method='linear', limit=limit)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # 처리 결과 로깅
            missing_after = data.isnull().sum()
            self.logger.info(f"Missing values before: {missing_before}")
            self.logger.info(f"Missing values after: {missing_after}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def detect_outliers(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        이상치 탐지 및 처리
        
        Args:
            data (pd.DataFrame): 입력 데이터
            columns (List[str], optional): 처리할 컬럼 목록
            method (str): 이상치 탐지 방법 ('zscore', 'iqr')
            threshold (float): 이상치 판단 임계값
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (이상치가 처리된 데이터, 이상치 마스크)
        """
        try:
            self.logger.info(f"Detecting outliers using {method} method")
            
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            # 이상치 마스크 초기화
            outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
            
            for column in columns:
                if method == 'zscore':
                    # Z-score 기반 이상치 탐지
                    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                    outliers = z_scores > threshold
                elif method == 'iqr':
                    # IQR 기반 이상치 탐지
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                outlier_mask[column] = outliers
                
                # 이상치 개수 로깅
                n_outliers = outliers.sum()
                self.logger.info(f"Found {n_outliers} outliers in column {column}")
            
            # 이상치 처리 (중앙값으로 대체)
            processed_data = data.copy()
            for column in columns:
                processed_data.loc[outlier_mask[column], column] = processed_data[column].median()
            
            return processed_data, outlier_mask
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {str(e)}")
            raise
    
    def normalize_data(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        데이터 정규화/표준화
        
        Args:
            data (pd.DataFrame): 입력 데이터
            columns (List[str], optional): 처리할 컬럼 목록
            method (str): 정규화 방법 ('minmax', 'zscore')
            
        Returns:
            pd.DataFrame: 정규화된 데이터
        """
        try:
            self.logger.info(f"Normalizing data using {method} method")
            
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            normalized_data = data.copy()
            
            for column in columns:
                if method == 'minmax':
                    # Min-Max 정규화
                    min_val = data[column].min()
                    max_val = data[column].max()
                    normalized_data[column] = (data[column] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    # Z-score 표준화
                    normalized_data[column] = (data[column] - data[column].mean()) / data[column].std()
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                # 통계 정보 로깅
                self.logger.info(f"Column {column} statistics after normalization:")
                self.logger.info(f"  Mean: {normalized_data[column].mean():.4f}")
                self.logger.info(f"  Std: {normalized_data[column].std():.4f}")
                self.logger.info(f"  Min: {normalized_data[column].min():.4f}")
                self.logger.info(f"  Max: {normalized_data[column].max():.4f}")
            
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Error normalizing data: {str(e)}")
            raise
    
    def extract_period(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        특정 기간 데이터 추출
        
        Args:
            data (pd.DataFrame): 입력 데이터
            start_date (datetime, optional): 시작 날짜
            end_date (datetime, optional): 종료 날짜
            period (str, optional): 기간 ('1d', '1w', '1m', '3m', '6m', '1y')
            
        Returns:
            pd.DataFrame: 추출된 데이터
        """
        try:
            self.logger.info("Extracting data for specific period")
            
            if period:
                # 기간 문자열을 datetime으로 변환
                end_date = datetime.now()
                if period == '1d':
                    start_date = end_date - timedelta(days=1)
                elif period == '1w':
                    start_date = end_date - timedelta(weeks=1)
                elif period == '1m':
                    start_date = end_date - timedelta(days=30)
                elif period == '3m':
                    start_date = end_date - timedelta(days=90)
                elif period == '6m':
                    start_date = end_date - timedelta(days=180)
                elif period == '1y':
                    start_date = end_date - timedelta(days=365)
                else:
                    raise ValueError(f"Unsupported period: {period}")
            
            # 날짜 필터링
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # 결과 로깅
            self.logger.info(f"Extracted data from {data.index.min()} to {data.index.max()}")
            self.logger.info(f"Total records: {len(data)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error extracting period data: {str(e)}")
            raise
    
    def process_data(
        self,
        data: pd.DataFrame,
        handle_missing: bool = True,
        detect_outliers: bool = True,
        normalize: bool = True,
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        전체 데이터 전처리 파이프라인
        
        Args:
            data (pd.DataFrame): 입력 데이터
            handle_missing (bool): 결측값 처리 여부
            detect_outliers (bool): 이상치 처리 여부
            normalize (bool): 정규화 여부
            period (str, optional): 추출할 기간
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        try:
            self.logger.info("Starting data preprocessing pipeline")
            
            # 데이터 복사
            processed_data = data.copy()
            
            # 결측값 처리
            if handle_missing:
                processed_data = self.handle_missing_values(processed_data)
            
            # 이상치 처리
            if detect_outliers:
                processed_data, _ = self.detect_outliers(processed_data)
            
            # 정규화
            if normalize:
                processed_data = self.normalize_data(processed_data)
            
            # 기간 추출
            if period:
                processed_data = self.extract_period(processed_data, period=period)
            
            self.logger.info("Data preprocessing completed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing pipeline: {str(e)}")
            raise 