"""
데이터 수집기 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import ccxt
import websocket
import json
import threading
import time
from datetime import datetime, timedelta
import logging
from src.utils.logger import get_logger
from src.utils.database import DatabaseManager

class DataCollector:
    """데이터 수집기 클래스"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        symbols: List[str] = ['BTC/USDT'],
        timeframes: List[str] = ['1h'],
        exchange_id: str = 'binance'
    ):
        """
        데이터 수집기 초기화
        
        Args:
            api_key (Optional[str]): API 키
            api_secret (Optional[str]): API 시크릿
            symbols (List[str]): 수집할 심볼 목록
            timeframes (List[str]): 수집할 타임프레임 목록
            exchange_id (str): 거래소 ID
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.timeframes = timeframes
        self.exchange_id = exchange_id
        
        self.logger = get_logger(__name__)
        self.db = DatabaseManager()
        
        # 거래소 초기화
        self.exchange = self._init_exchange()
        
        # WebSocket 연결 상태
        self.ws_connected = False
        self.ws_thread = None
        self.ws_callbacks = {}
        
    def _init_exchange(self) -> ccxt.Exchange:
        """
        거래소 초기화
        
        Returns:
            ccxt.Exchange: 거래소 객체
        """
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True
            })
            return exchange
            
        except Exception as e:
            self.logger.error(f"거래소 초기화 실패: {str(e)}")
            raise
            
    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        과거 데이터 수집
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 타임프레임
            start_time (datetime): 시작 시간
            end_time (datetime): 종료 시간
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            self.logger.info(f"과거 데이터 수집 시작: {symbol} {timeframe}")
            
            # 데이터베이스에서 기존 데이터 확인
            existing_data = self.db.get_market_data(symbol, timeframe, start_time, end_time)
            if not existing_data.empty:
                self.logger.info("기존 데이터 사용")
                return existing_data
                
            # 데이터 수집
            data = []
            current_time = start_time
            
            while current_time < end_time:
                try:
                    # API 요청 제한 관리
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                    # 데이터 수집
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe,
                        since=int(current_time.timestamp() * 1000),
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                        
                    data.extend(ohlcv)
                    current_time = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                    
                except Exception as e:
                    self.logger.error(f"데이터 수집 중 오류 발생: {str(e)}")
                    time.sleep(5)
                    continue
                    
            # 데이터프레임 생성
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 데이터베이스에 저장
            self.db.save_market_data(df, symbol, timeframe)
            
            self.logger.info(f"과거 데이터 수집 완료: {len(df)}개")
            return df
            
        except Exception as e:
            self.logger.error(f"과거 데이터 수집 실패: {str(e)}")
            raise
            
    def start_websocket_stream(
        self,
        callback_function: Callable[[Dict[str, Any]], None]
    ):
        """
        WebSocket 스트림 시작
        
        Args:
            callback_function (Callable[[Dict[str, Any]], None]): 콜백 함수
        """
        try:
            # WebSocket URL 생성
            ws_url = f"wss://stream.binance.com:9443/ws"
            
            # WebSocket 연결
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # 콜백 함수 저장
            self.ws_callbacks['market_data'] = callback_function
            
            # WebSocket 스레드 시작
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # 구독 메시지 전송
            self._subscribe_streams()
            
        except Exception as e:
            self.logger.error(f"WebSocket 스트림 시작 실패: {str(e)}")
            raise
            
    def _on_message(self, ws, message):
        """
        WebSocket 메시지 처리
        
        Args:
            ws: WebSocket 객체
            message: 수신 메시지
        """
        try:
            data = json.loads(message)
            
            # 콜백 함수 호출
            if 'market_data' in self.ws_callbacks:
                self.ws_callbacks['market_data'](data)
                
        except Exception as e:
            self.logger.error(f"WebSocket 메시지 처리 실패: {str(e)}")
            
    def _on_error(self, ws, error):
        """
        WebSocket 에러 처리
        
        Args:
            ws: WebSocket 객체
            error: 에러 객체
        """
        self.logger.error(f"WebSocket 에러: {str(error)}")
        self.ws_connected = False
        
    def _on_close(self, ws, close_status_code, close_msg):
        """
        WebSocket 연결 종료 처리
        
        Args:
            ws: WebSocket 객체
            close_status_code: 종료 상태 코드
            close_msg: 종료 메시지
        """
        self.logger.info("WebSocket 연결 종료")
        self.ws_connected = False
        
    def _on_open(self, ws):
        """
        WebSocket 연결 시작 처리
        
        Args:
            ws: WebSocket 객체
        """
        self.logger.info("WebSocket 연결 시작")
        self.ws_connected = True
        
    def _subscribe_streams(self):
        """스트림 구독"""
        try:
            # 구독 메시지 생성
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [
                    f"{symbol.lower()}@kline_{timeframe}"
                    for symbol in self.symbols
                    for timeframe in self.timeframes
                ],
                "id": 1
            }
            
            # 구독 메시지 전송
            self.ws.send(json.dumps(subscribe_message))
            
        except Exception as e:
            self.logger.error(f"스트림 구독 실패: {str(e)}")
            raise
            
    def save_data(
        self,
        data: pd.DataFrame,
        file_path: Optional[str] = None,
        db_connection: Optional[Any] = None
    ):
        """
        데이터 저장
        
        Args:
            data (pd.DataFrame): 저장할 데이터
            file_path (Optional[str]): 파일 경로
            db_connection (Optional[Any]): 데이터베이스 연결
        """
        try:
            if file_path:
                data.to_csv(file_path, index=False)
                self.logger.info(f"데이터를 파일에 저장: {file_path}")
                
            if db_connection:
                data.to_sql('market_data', db_connection, if_exists='append', index=False)
                self.logger.info("데이터를 데이터베이스에 저장")
                
        except Exception as e:
            self.logger.error(f"데이터 저장 실패: {str(e)}")
            raise
            
    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        최신 데이터 조회
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 타임프레임
            limit (int): 조회 개수
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            # 데이터베이스에서 조회
            data = self.db.get_latest_market_data(symbol, timeframe, limit)
            
            if data.empty:
                # API에서 조회
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                data = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                
                # 데이터베이스에 저장
                self.db.save_market_data(data, symbol, timeframe)
                
            return data
            
        except Exception as e:
            self.logger.error(f"최신 데이터 조회 실패: {str(e)}")
            raise 