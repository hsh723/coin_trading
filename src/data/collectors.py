"""
바이낸스 데이터 수집 모듈
"""

import ccxt
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from ..utils.logger import setup_logger
from ..utils.config_loader import get_config

class BinanceDataCollector:
    """
    바이낸스 거래소에서 데이터를 수집하는 클래스
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        초기화
        
        Args:
            api_key (str): 바이낸스 API 키
            api_secret (str): 바이낸스 API 시크릿
            testnet (bool): 테스트넷 사용 여부
        """
        self.logger = setup_logger()
        self.config = get_config()['config']
        
        # ccxt 바이낸스 객체 초기화
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'testnet': testnet
            }
        })
        
        self.logger.info(f"BinanceDataCollector initialized (testnet: {testnet})")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 수집
        
        Args:
            symbol (str): 거래 심볼 (예: 'BTC/USDT')
            timeframe (str): 시간 프레임 (예: '1h', '4h', '1d')
            since (int, optional): 시작 시간 (밀리초)
            limit (int): 수집할 데이터 개수
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            self.logger.info(f"Fetching OHLCV data for {symbol} ({timeframe})")
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Successfully fetched {len(df)} OHLCV records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise
    
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        과거 데이터 수집
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            start_date (datetime, optional): 시작 날짜
            end_date (datetime, optional): 종료 날짜
            
        Returns:
            pd.DataFrame: 과거 데이터
        """
        try:
            self.logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            # 시작 시간을 밀리초로 변환
            since = int(start_date.timestamp() * 1000)
            
            # 데이터 수집
            all_data = []
            while since < end_date.timestamp() * 1000:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                self.logger.info(f"Fetched data until {datetime.fromtimestamp(since/1000)}")
            
            # DataFrame 생성
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 종료 날짜까지 필터링
            df = df[df.index <= end_date]
            
            self.logger.info(f"Successfully fetched {len(df)} historical records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        현재 시세 정보 수집
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Dict: 현재 시세 정보
        """
        try:
            self.logger.info(f"Fetching ticker data for {symbol}")
            
            ticker = self.exchange.fetch_ticker(symbol)
            
            # 필요한 정보만 추출
            result = {
                'symbol': ticker['symbol'],
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'high': ticker['high'],
                'low': ticker['low'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
            }
            
            self.logger.info(f"Successfully fetched ticker data for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker data: {str(e)}")
            raise
    
    def fetch_funding_rate(self, symbol: str) -> Dict:
        """
        자금 비율 수집
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Dict: 자금 비율 정보
        """
        try:
            self.logger.info(f"Fetching funding rate for {symbol}")
            
            funding_rate = self.exchange.fetch_funding_rate(symbol)
            
            result = {
                'symbol': funding_rate['symbol'],
                'funding_rate': funding_rate['fundingRate'],
                'next_funding_time': datetime.fromtimestamp(funding_rate['nextFundingTime'] / 1000),
                'timestamp': datetime.fromtimestamp(funding_rate['timestamp'] / 1000)
            }
            
            self.logger.info(f"Successfully fetched funding rate for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rate: {str(e)}")
            raise
    
    def __del__(self):
        """소멸자"""
        try:
            self.exchange.close()
            self.logger.info("BinanceDataCollector 종료")
        except Exception as e:
            self.logger.error(f"BinanceDataCollector 종료 중 오류 발생: {str(e)}") 