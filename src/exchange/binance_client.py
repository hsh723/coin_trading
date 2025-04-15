from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from datetime import datetime
import time

class BinanceClient:
    """바이낸스 API 클라이언트"""
    
    def __init__(self, api_key=None, api_secret=None, test_mode=False):
        """바이낸스 클라이언트 초기화

        Args:
            api_key (str): API 키
            api_secret (str): API 시크릿
            test_mode (bool, optional): 테스트 모드 여부. Defaults to False.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
        self.client = None
        self.time_offset = 0
        self.logger = logging.getLogger(__name__)
        
        # 테스트 모드에서 사용할 모의 데이터
        self.mock_data = {
            'BTCUSDT': {
                'price': 50000.0,
                'bid': 49999.0,
                'ask': 50001.0,
                'volume': 1000.0,
                'price_change': 1.5
            },
            'balances': {
                'BTC': {'free': 1.0, 'locked': 0.0},
                'USDT': {'free': 50000.0, 'locked': 0.0}
            },
            'orders': []
        }
        
    async def _sync_time(self) -> bool:
        """서버 시간과 동기화"""
        try:
            # 서버 시간 조회
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            
            # 시간 차이 계산 (서버 시간 - 로컬 시간)
            time_diff = server_time['serverTime'] - local_time
            
            # 시간 차이가 1000ms 이상이면 보정
            if abs(time_diff) > 1000:
                # 시간 차이가 음수면 양수로 변환하여 적용
                if time_diff < 0:
                    self.time_offset = abs(time_diff) + 2000  # 2초 여유를 더함
                else:
                    self.time_offset = time_diff + 1000  # 1초 여유를 더함
                self.logger.warning(f"시간 차이 보정: {time_diff}ms, 적용된 오프셋: {self.time_offset}ms")
            else:
                self.time_offset = 0
            
            self.logger.info(f"시간 동기화 완료: 오프셋={self.time_offset}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"시간 동기화 실패: {str(e)}")
            return False

    def _get_timestamp(self) -> int:
        """동기화된 타임스탬프 생성"""
        timestamp = int(time.time() * 1000) + self.time_offset
        self.logger.debug(f"생성된 타임스탬프: {timestamp}")
        return timestamp
            
    async def initialize(self) -> bool:
        """바이낸스 클라이언트 초기화"""
        try:
            # 서버 시간 동기화
            server_time = await self.get_server_time()
            time_diff = int(time.time() * 1000) - server_time['serverTime']
            
            if abs(time_diff) > 1000:
                self.logger.warning(f"서버 시간과 로컬 시간의 차이가 {time_diff}ms 입니다")
                return False
            
            # 계정 정보 확인
            account = await self.get_account()
            if not account or 'accountType' not in account:
                self.logger.error("계정 정보를 가져올 수 없습니다")
                return False
            
            self.logger.info("바이낸스 클라이언트가 성공적으로 초기화되었습니다")
            return True
            
        except Exception as e:
            self.logger.error(f"바이낸스 클라이언트 초기화 중 오류 발생: {str(e)}")
            return False

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """시장 데이터 조회

        Args:
            symbol (str): 거래 쌍

        Returns:
            Dict[str, Any]: 시장 데이터
        """
        try:
            if self.test_mode:
                return {
                    'price': 50000.0,
                    'bid': 49990.0,
                    'ask': 50010.0,
                    'volume': 100.0,
                    'price_change_24h': -1.5
                }
            
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            depth = self.client.get_order_book(symbol=symbol, limit=5)
            ticker_24h = self.client.get_ticker(symbol=symbol)
            
            return {
                'price': float(ticker['price']),
                'bid': float(depth['bids'][0][0]),
                'ask': float(depth['asks'][0][0]),
                'volume': float(ticker_24h['volume']),
                'price_change_24h': float(ticker_24h['priceChangePercent'])
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 조회 실패: {str(e)}")
            raise

    def create_order(self, **params) -> Dict[str, Any]:
        """주문 생성

        Args:
            **params: 주문 파라미터

        Returns:
            Dict[str, Any]: 주문 결과
        """
        try:
            if self.test_mode:
                return {
                    'orderId': '12345',
                    'symbol': params.get('symbol'),
                    'side': params.get('side'),
                    'type': params.get('type'),
                    'origQty': str(params.get('quantity')),
                    'status': 'NEW'
                }
            
            return self.client.create_order(**params)
            
        except Exception as e:
            self.logger.error(f"주문 생성 실패: {str(e)}")
            raise

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """주문 취소

        Args:
            symbol (str): 거래 쌍
            order_id (str): 주문 ID

        Returns:
            Dict[str, Any]: 취소 결과
        """
        try:
            if self.test_mode:
                return {
                    'orderId': order_id,
                    'status': 'CANCELED'
                }
            
            return self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {str(e)}")
            raise

    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """주문 정보 조회

        Args:
            symbol (str): 거래 쌍
            order_id (str): 주문 ID

        Returns:
            Dict[str, Any]: 주문 정보
        """
        try:
            if self.test_mode:
                return {
                    'orderId': order_id,
                    'symbol': symbol,
                    'side': 'BUY',
                    'type': 'MARKET',
                    'origQty': '0.001',
                    'status': 'FILLED',
                    'transactTime': int(time.time() * 1000)
                }
            
            return self.client.get_order(
                symbol=symbol,
                orderId=order_id
            )
            
        except Exception as e:
            self.logger.error(f"주문 정보 조회 실패: {str(e)}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """미체결 주문 조회

        Args:
            symbol (Optional[str], optional): 거래 쌍. Defaults to None.

        Returns:
            list: 미체결 주문 목록
        """
        try:
            if self.test_mode:
                return []
            
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            return self.client.get_open_orders()
            
        except Exception as e:
            self.logger.error(f"미체결 주문 조회 실패: {str(e)}")
            raise

    def get_account_balance(self) -> Dict[str, Dict[str, float]]:
        """계정 잔고 조회

        Returns:
            Dict[str, Dict[str, float]]: 자산별 잔고
        """
        try:
            if self.test_mode:
                return {
                    'BTC': {
                        'free': 1.0,
                        'locked': 0.0
                    },
                    'USDT': {
                        'free': 50000.0,
                        'locked': 0.0
                    }
                }
            
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked
                    }
            
            return balances
            
        except Exception as e:
            self.logger.error(f"계정 잔고 조회 실패: {str(e)}")
            raise 