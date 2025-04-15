"""
바이낸스 거래소 모듈
"""

import ccxt.async_support as ccxt
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.utils.logger import setup_logger
import pandas as pd
from .base import ExchangeBase
import logging

logger = logging.getLogger(__name__)

class BinanceExchange(ExchangeBase):
    """
    바이낸스 거래소 클래스
    바이낸스 API를 통한 거래 기능 제공
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ):
        """
        거래소 초기화
        
        Args:
            api_key (str): API 키
            api_secret (str): API 시크릿
            testnet (bool): 테스트넷 사용 여부
        """
        self.logger = logger
        self.testnet = testnet
        
        # 거래소 객체 생성
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'testnet': testnet
            }
        })
        
    async def initialize(self):
        """
        거래소 초기화
        """
        try:
            await self.exchange.load_markets()
            self.logger.info("거래소 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"거래소 초기화 실패: {str(e)}")
            raise
            
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 조회
        
        Args:
            symbol (str): 심볼
            timeframe (str): 시간 프레임
            limit (int): 데이터 개수
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            self.handle_error(e)
            
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        현재가 조회
        
        Args:
            symbol (str): 심볼
            
        Returns:
            Optional[Dict[str, Any]]: 현재가 정보
        """
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            self.logger.error(f"현재가 조회 실패: {str(e)}")
            return None
            
    async def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        잔고 조회
        
        Returns:
            Optional[Dict[str, Any]]: 잔고 정보
        """
        try:
            balance = await self.exchange.fetch_balance()
            return balance
            
        except Exception as e:
            self.logger.error(f"잔고 조회 실패: {str(e)}")
            return None
            
    async def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        주문 생성
        
        Args:
            symbol (str): 심볼
            type (str): 주문 유형
            side (str): 주문 방향
            amount (float): 주문 수량
            price (Optional[float]): 주문 가격
            
        Returns:
            Optional[Dict[str, Any]]: 주문 정보
        """
        try:
            order = await self.exchange.create_order(
                symbol,
                type,
                side,
                amount,
                price
            )
            return order
            
        except Exception as e:
            self.handle_error(e)
            
    async def cancel_order(
        self,
        id: str,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        주문 취소
        
        Args:
            id (str): 주문 ID
            symbol (str): 심볼
            
        Returns:
            Optional[Dict[str, Any]]: 취소된 주문 정보
        """
        try:
            order = await self.exchange.cancel_order(id, symbol)
            return order
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {str(e)}")
            return None
            
    async def fetch_order(
        self,
        id: str,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        주문 조회
        
        Args:
            id (str): 주문 ID
            symbol (str): 심볼
            
        Returns:
            Optional[Dict[str, Any]]: 주문 정보
        """
        try:
            order = await self.exchange.fetch_order(id, symbol)
            return order
            
        except Exception as e:
            self.logger.error(f"주문 조회 실패: {str(e)}")
            return None
            
    async def fetch_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        미체결 주문 조회
        
        Args:
            symbol (Optional[str]): 심볼
            
        Returns:
            List[Dict[str, Any]]: 미체결 주문 목록
        """
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            return orders
            
        except Exception as e:
            self.logger.error(f"미체결 주문 조회 실패: {str(e)}")
            return []
            
    async def fetch_closed_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        체결 주문 조회
        
        Args:
            symbol (Optional[str]): 심볼
            
        Returns:
            List[Dict[str, Any]]: 체결 주문 목록
        """
        try:
            orders = await self.exchange.fetch_closed_orders(symbol)
            return orders
            
        except Exception as e:
            self.logger.error(f"체결 주문 조회 실패: {str(e)}")
            return []
            
    async def close(self):
        """
        거래소 연결 종료
        """
        try:
            await self.exchange.close()
            self.logger.info("거래소 연결 종료")
            
        except Exception as e:
            self.logger.error(f"거래소 연결 종료 실패: {str(e)}")
            raise

    def handle_error(self, error: Exception) -> None:
        """에러 처리"""
        # 구현...