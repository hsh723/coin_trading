"""
데이터 처리기 모듈
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    데이터 처리기 클래스
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시장 데이터 처리
        
        Args:
            df (pd.DataFrame): 원본 시장 데이터
            
        Returns:
            pd.DataFrame: 처리된 시장 데이터
        """
        try:
            # 데이터 정렬
            df = df.sort_values('timestamp')
            
            # 결측치 처리
            df = df.dropna()
            
            # 중복 제거
            df = df.drop_duplicates()
            
            # 이상치 처리
            df = self._remove_outliers(df)
            
            # 기술적 지표 계산
            df = self._calculate_technical_indicators(df)
            
            self.logger.info("시장 데이터 처리 완료")
            return df
            
        except Exception as e:
            self.logger.error(f"시장 데이터 처리 실패: {str(e)}")
            raise
            
    def process_trade_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래 데이터 처리
        
        Args:
            df (pd.DataFrame): 원본 거래 데이터
            
        Returns:
            pd.DataFrame: 처리된 거래 데이터
        """
        try:
            # 데이터 정렬
            df = df.sort_values('timestamp')
            
            # 결측치 처리
            df = df.dropna()
            
            # 중복 제거
            df = df.drop_duplicates()
            
            # 거래량 가중 평균 가격 계산
            df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            self.logger.info("거래 데이터 처리 완료")
            return df
            
        except Exception as e:
            self.logger.error(f"거래 데이터 처리 실패: {str(e)}")
            raise
            
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 제거
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 이상치가 제거된 데이터프레임
        """
        try:
            # Z-score 방식으로 이상치 제거
            for column in ['open', 'high', 'low', 'close', 'volume']:
                if column in df.columns:
                    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                    df = df[z_scores < 3]
            
            return df
            
        except Exception as e:
            self.logger.error(f"이상치 제거 실패: {str(e)}")
            raise
            
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 계산
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터프레임
        """
        try:
            # 이동평균선
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # 거래량 이동평균
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            
            # 상대강도지수 (RSI)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
            df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {str(e)}")
            raise

# 전역 인스턴스 생성
data_processor = DataProcessor() 