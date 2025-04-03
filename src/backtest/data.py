"""
백테스트 데이터 관리 모듈

이 모듈은 백테스트에 필요한 과거 시장 데이터를 관리합니다.
주요 기능:
- 과거 시장 데이터 다운로드
- 데이터 전처리 및 정규화
- 다양한 시간대 데이터 관리
- 데이터 캐싱
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
import json

# 로거 설정
logger = logging.getLogger(__name__)

class BacktestData:
    """백테스트 데이터 관리 클래스"""
    
    def __init__(self, exchange_name: str, config: Dict):
        """
        백테스트 데이터 관리자 초기화
        
        Args:
            exchange_name (str): 거래소 이름
            config (Dict): 설정 정보
        """
        self.exchange_name = exchange_name
        self.config = config
        self.exchange = None
        self.data_dir = Path("data/backtest")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 캐시
        self.cache = {}
        
    async def initialize(self):
        """데이터 관리자 초기화"""
        try:
            # 거래소 초기화
            self.exchange = getattr(ccxt, self.exchange_name)({
                'enableRateLimit': True,
                'timeout': self.config['exchange']['timeout']
            })
            
            logger.info(f"백테스트 데이터 관리자 초기화 완료 ({self.exchange_name})")
            
        except Exception as e:
            logger.error(f"데이터 관리자 초기화 실패: {str(e)}")
            raise
            
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        과거 시장 데이터 조회
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            start_date (datetime): 시작 날짜
            end_date (datetime): 종료 날짜
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            # 캐시 키 생성
            cache_key = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            # 캐시된 데이터 확인
            if cache_key in self.cache:
                logger.info(f"캐시된 데이터 사용: {cache_key}")
                return self.cache[cache_key]
                
            # 파일 캐시 확인
            cache_file = self.data_dir / f"{cache_key}.csv"
            if cache_file.exists():
                logger.info(f"파일 캐시에서 데이터 로드: {cache_file}")
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.cache[cache_key] = df
                return df
                
            # 데이터 다운로드
            logger.info(f"과거 데이터 다운로드 시작: {symbol} {timeframe}")
            
            # 시간 프레임별 데이터 개수 계산
            timeframe_minutes = self._get_timeframe_minutes(timeframe)
            total_minutes = int((end_date - start_date).total_seconds() / 60)
            limit = min(total_minutes // timeframe_minutes, 1000)  # 거래소 제한 고려
            
            # 데이터 조회
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=int(start_date.timestamp() * 1000),
                limit=limit
            )
            
            # DataFrame 변환
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 기술적 지표 추가
            df = self._add_technical_indicators(df)
            
            # 데이터 저장
            df.to_csv(cache_file, index=False)
            self.cache[cache_key] = df
            
            logger.info(f"과거 데이터 다운로드 완료: {len(df)} 개")
            return df
            
        except Exception as e:
            logger.error(f"과거 데이터 조회 실패: {str(e)}")
            raise
            
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """
        시간 프레임을 분 단위로 변환
        
        Args:
            timeframe (str): 시간 프레임 (예: '1m', '5m', '1h', '1d')
            
        Returns:
            int: 분 단위 시간
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        else:
            raise ValueError(f"지원하지 않는 시간 프레임: {timeframe}")
            
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터
        """
        # 이동평균선
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 볼린저 밴드
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
        
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        데이터 요약 정보 조회
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            Dict: 데이터 요약 정보
        """
        return {
            'start_date': df['timestamp'].min(),
            'end_date': df['timestamp'].max(),
            'total_candles': len(df),
            'price_range': {
                'min': df['low'].min(),
                'max': df['high'].max(),
                'current': df['close'].iloc[-1]
            },
            'volume': {
                'total': df['volume'].sum(),
                'average': df['volume'].mean()
            }
        }
        
    def cleanup_cache(self):
        """데이터 캐시 정리"""
        self.cache.clear()
        
    async def close(self):
        """리소스 정리"""
        if self.exchange:
            await self.exchange.close() 