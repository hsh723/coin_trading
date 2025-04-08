"""
데이터 수집기 모듈
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class DataCollector:
    """
    데이터 수집기 클래스
    """
    def __init__(self, exchange_id: str = 'binance'):
        """
        데이터 수집기 초기화
        
        Args:
            exchange_id (str): 거래소 ID
        """
        self.exchange = getattr(ccxt, exchange_id)()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """로깅 설정"""
        self.logger.setLevel(logging.INFO)
        
    async def collect_market_data(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        시장 데이터 수집
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간대
            start_date (datetime): 시작 날짜
            end_date (datetime): 종료 날짜
            
        Returns:
            pd.DataFrame: 수집된 시장 데이터
        """
        try:
            # 임시 데이터 생성 (실제 구현에서는 거래소 API 사용)
            dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
            data = {
                'timestamp': dates,
                'open': np.random.normal(50000, 1000, len(dates)),
                'high': np.random.normal(51000, 1000, len(dates)),
                'low': np.random.normal(49000, 1000, len(dates)),
                'close': np.random.normal(50000, 1000, len(dates)),
                'volume': np.random.normal(1000, 100, len(dates))
            }
            df = pd.DataFrame(data)
            
            self.logger.info(f"시장 데이터 수집 완료: {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패: {str(e)}")
            raise
            
    async def collect_trade_data(self, symbol: str, 
                                start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        거래 데이터 수집
        
        Args:
            symbol (str): 거래 심볼
            start_date (datetime): 시작 날짜
            end_date (datetime): 종료 날짜
            
        Returns:
            pd.DataFrame: 수집된 거래 데이터
        """
        try:
            # 임시 데이터 생성 (실제 구현에서는 거래소 API 사용)
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            data = {
                'timestamp': dates,
                'price': np.random.normal(50000, 1000, len(dates)),
                'volume': np.random.normal(1000, 100, len(dates)),
                'side': np.random.choice(['buy', 'sell'], len(dates))
            }
            df = pd.DataFrame(data)
            
            self.logger.info(f"거래 데이터 수집 완료: {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"거래 데이터 수집 실패: {str(e)}")
            raise

    def get_historical_data(
        self,
        symbol: str,
        start_time: str,
        end_time: str,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        과거 데이터 수집
        
        Args:
            symbol (str): 거래 심볼
            start_time (str): 시작 시간 (YYYY-MM-DD)
            end_time (str): 종료 시간 (YYYY-MM-DD)
            interval (str): 시간 간격
            
        Returns:
            pd.DataFrame: 과거 데이터
        """
        try:
            # 시간 변환
            start_timestamp = int(datetime.strptime(start_time, '%Y-%m-%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_time, '%Y-%m-%d').timestamp() * 1000)
            
            # 시간 간격 변환
            timeframe = self._convert_interval(interval)
            
            # 데이터 수집
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=start_timestamp,
                limit=None
            )
            
            # 데이터프레임 변환
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 인덱스 설정
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 종료 시간 이후 데이터 제거
            df = df[df.index <= pd.to_datetime(end_time)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"과거 데이터 수집 실패: {str(e)}")
            raise
            
    def get_realtime_data(
        self,
        symbol: str,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        실시간 데이터 수집
        
        Args:
            symbol (str): 거래 심볼
            interval (str): 시간 간격
            
        Returns:
            pd.DataFrame: 실시간 데이터
        """
        try:
            # 시간 간격 변환
            timeframe = self._convert_interval(interval)
            
            # 데이터 수집
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=1
            )
            
            # 데이터프레임 변환
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 인덱스 설정
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"실시간 데이터 수집 실패: {str(e)}")
            raise
            
    def _convert_interval(self, interval: str) -> str:
        """
        시간 간격 변환
        
        Args:
            interval (str): 시간 간격
            
        Returns:
            str: 변환된 시간 간격
        """
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        return interval_map.get(interval, '1h')

# 전역 인스턴스 생성
data_collector = DataCollector() 