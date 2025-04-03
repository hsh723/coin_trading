"""
거래소 API 래퍼 모듈
"""

import ccxt
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from ..utils.logger import setup_logger
from ..utils.config_loader import get_config

class BinanceExchange:
    """
    바이낸스 거래소 API 래퍼 클래스
    """
    
    def __init__(self):
        """
        거래소 API 초기화
        """
        # 설정 로드
        self.config = get_config()['config']['exchange']
        
        # API 키 설정
        self.api_key = self.config['api_key']
        self.api_secret = self.config['api_secret']
        
        # ccxt 초기화
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 선물 거래
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        })
        
        # 재시도 설정
        self.max_retries = self.config['max_retries']
        self.retry_delay = self.config['retry_delay']
        
        # 로거 설정
        self.logger = setup_logger()
        self.logger.info("BinanceExchange initialized")
    
    def _retry_on_error(self, func, *args, **kwargs) -> Any:
        """
        에러 발생 시 재시도하는 데코레이터 함수
        
        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
            
        Returns:
            Any: 함수 실행 결과
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Max retries reached: {str(e)}")
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        계정 정보 조회
        
        Returns:
            Dict[str, Any]: 계정 정보
        """
        try:
            account_info = self._retry_on_error(self.exchange.fetch_balance)
            
            # 필요한 정보만 추출
            result = {
                'total_balance': account_info['total'],
                'used_balance': account_info['used'],
                'free_balance': account_info['free'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Account info fetched: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching account info: {str(e)}")
            raise
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        시장가 주문 실행
        
        Args:
            symbol (str): 거래 심볼
            side (str): 주문 방향 ('buy' 또는 'sell')
            amount (float): 주문 수량
            reduce_only (bool): 감소 전용 주문 여부
            
        Returns:
            Dict[str, Any]: 주문 정보
        """
        try:
            order = self._retry_on_error(
                self.exchange.create_order,
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params={'reduceOnly': reduce_only}
            )
            
            self.logger.info(
                f"Market order placed: {symbol} {side} {amount} "
                f"(order_id: {order['id']})"
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing market order: {str(e)}")
            raise
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
        post_only: bool = False
    ) -> Dict[str, Any]:
        """
        지정가 주문 실행
        
        Args:
            symbol (str): 거래 심볼
            side (str): 주문 방향 ('buy' 또는 'sell')
            amount (float): 주문 수량
            price (float): 주문 가격
            reduce_only (bool): 감소 전용 주문 여부
            post_only (bool): 메이커 전용 주문 여부
            
        Returns:
            Dict[str, Any]: 주문 정보
        """
        try:
            order = self._retry_on_error(
                self.exchange.create_order,
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params={
                    'reduceOnly': reduce_only,
                    'postOnly': post_only
                }
            )
            
            self.logger.info(
                f"Limit order placed: {symbol} {side} {amount} @ {price} "
                f"(order_id: {order['id']})"
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing limit order: {str(e)}")
            raise
    
    def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> Dict[str, Any]:
        """
        주문 취소
        
        Args:
            symbol (str): 거래 심볼
            order_id (str): 주문 ID
            
        Returns:
            Dict[str, Any]: 취소된 주문 정보
        """
        try:
            result = self._retry_on_error(
                self.exchange.cancel_order,
                order_id,
                symbol
            )
            
            self.logger.info(f"Order cancelled: {symbol} {order_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {str(e)}")
            raise
    
    def modify_order(
        self,
        symbol: str,
        order_id: str,
        new_amount: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        주문 수정
        
        Args:
            symbol (str): 거래 심볼
            order_id (str): 주문 ID
            new_amount (float, optional): 새로운 주문 수량
            new_price (float, optional): 새로운 주문 가격
            
        Returns:
            Dict[str, Any]: 수정된 주문 정보
        """
        try:
            # 기존 주문 정보 조회
            order = self._retry_on_error(
                self.exchange.fetch_order,
                order_id,
                symbol
            )
            
            # 주문 취소
            self.cancel_order(symbol, order_id)
            
            # 새로운 주문 생성
            if new_amount is None:
                new_amount = order['amount']
            if new_price is None:
                new_price = order['price']
            
            new_order = self.place_limit_order(
                symbol=symbol,
                side=order['side'],
                amount=new_amount,
                price=new_price,
                reduce_only=order['reduceOnly'],
                post_only=order['postOnly']
            )
            
            self.logger.info(
                f"Order modified: {symbol} {order_id} -> {new_order['id']} "
                f"(amount: {new_amount}, price: {new_price})"
            )
            
            return new_order
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
            raise
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        포지션 정보 조회
        
        Args:
            symbol (str, optional): 거래 심볼 (None이면 전체 조회)
            
        Returns:
            List[Dict[str, Any]]: 포지션 정보 목록
        """
        try:
            positions = self._retry_on_error(self.exchange.fetch_positions)
            
            # 필요한 정보만 추출
            result = []
            for pos in positions:
                if pos['contracts'] > 0:  # 포지션이 있는 경우만
                    if symbol is None or pos['symbol'] == symbol:
                        result.append({
                            'symbol': pos['symbol'],
                            'side': pos['side'],
                            'size': pos['contracts'],
                            'entry_price': pos['entryPrice'],
                            'leverage': pos['leverage'],
                            'unrealized_pnl': pos['unrealizedPnl'],
                            'timestamp': datetime.now().isoformat()
                        })
            
            self.logger.info(f"Positions fetched: {len(result)} positions")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching positions: {str(e)}")
            raise
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        호가 정보 조회
        
        Args:
            symbol (str): 거래 심볼
            limit (int): 호가 개수
            
        Returns:
            Dict[str, Any]: 호가 정보
        """
        try:
            order_book = self._retry_on_error(
                self.exchange.fetch_order_book,
                symbol,
                limit
            )
            
            result = {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': order_book['timestamp']
            }
            
            self.logger.info(f"Order book fetched for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching order book: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        현재가 정보 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Dict[str, Any]: 현재가 정보
        """
        try:
            ticker = self._retry_on_error(
                self.exchange.fetch_ticker,
                symbol
            )
            
            result = {
                'symbol': ticker['symbol'],
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp']
            }
            
            self.logger.info(f"Ticker fetched for {symbol}: {result['last']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {str(e)}")
            raise 