import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    데이터 전처리 시스템
    
    주요 기능:
    - 데이터 정제 및 정규화
    - 기술적 지표 계산
    - 시계열 특성 추출
    - 데이터 품질 검증
    """
    
    def __init__(self,
                 data_dir: str = "./data",
                 save_dir: str = "./processed_data"):
        """
        데이터 프로세서 초기화
        
        Args:
            data_dir: 원본 데이터 디렉토리
            save_dir: 처리된 데이터 저장 디렉토리
        """
        self.data_dir = data_dir
        self.save_dir = save_dir
        
        # 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 메트릭스
        self.metrics = {
            'files_processed': 0,
            'rows_processed': 0,
            'errors': 0
        }
    
    def process_data(self, symbol: str) -> pd.DataFrame:
        """
        데이터 처리
        
        Args:
            symbol: 처리할 심볼
            
        Returns:
            처리된 데이터프레임
        """
        try:
            # 데이터 파일 찾기
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith(symbol)]
            if not data_files:
                logger.warning(f"데이터 파일을 찾을 수 없음: {symbol}")
                return pd.DataFrame()
            
            # 데이터 로드 및 병합
            dfs = []
            for file in data_files:
                df = pd.read_csv(os.path.join(self.data_dir, file))
                dfs.append(df)
            
            if not dfs:
                return pd.DataFrame()
            
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values('timestamp')
            
            # 데이터 정제
            df = self._clean_data(df)
            
            # 기술적 지표 계산
            df = self._calculate_indicators(df)
            
            # 시계열 특성 추출
            df = self._extract_features(df)
            
            # 데이터 저장
            filename = f"{self.save_dir}/{symbol}_processed_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"처리된 데이터 저장 완료: {filename}")
            
            # 메트릭스 업데이트
            self.metrics['files_processed'] += 1
            self.metrics['rows_processed'] += len(df)
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {e}")
            self.metrics['errors'] += 1
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제"""
        try:
            # 중복 제거
            df = df.drop_duplicates()
            
            # 결측값 처리
            df = df.fillna(method='ffill')
            
            # 이상치 처리
            for col in ['price', 'quantity']:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower_bound, upper_bound)
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 정제 중 오류 발생: {e}")
            return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # 이동평균선
            df['MA5'] = df['price'].rolling(window=5).mean()
            df['MA20'] = df['price'].rolling(window=20).mean()
            df['MA60'] = df['price'].rolling(window=60).mean()
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            df['BB_middle'] = df['price'].rolling(window=20).mean()
            df['BB_std'] = df['price'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
            
            # MACD
            exp1 = df['price'].ewm(span=12, adjust=False).mean()
            exp2 = df['price'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"지표 계산 중 오류 발생: {e}")
            return df
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시계열 특성 추출"""
        try:
            # 수익률
            df['returns'] = df['price'].pct_change()
            
            # 변동성
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # 거래량 특성
            df['volume_ma'] = df['quantity'].rolling(window=20).mean()
            df['volume_std'] = df['quantity'].rolling(window=20).std()
            
            # 가격 패턴
            df['price_range'] = df['price'].rolling(window=20).max() - df['price'].rolling(window=20).min()
            df['price_momentum'] = df['price'].pct_change(periods=5)
            
            return df
            
        except Exception as e:
            logger.error(f"특성 추출 중 오류 발생: {e}")
            return df
    
    def get_metrics(self) -> Dict[str, Any]:
        """처리 메트릭스 반환"""
        return self.metrics 