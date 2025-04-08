from typing import Dict, List, Optional
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BinanceAPI:
    """바이낸스 API 클라이언트"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        self.logger = logger
        
    def get_historical_klines(self, 
                            symbol: str, 
                            interval: str, 
                            limit: int = 1000) -> List[Dict]:
        """과거 K선 데이터 조회"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except BinanceAPIException as e:
            self.logger.error(f"K선 데이터 조회 실패: {e}")
            return []
            
    def create_market_order(self, 
                          symbol: str, 
                          side: str, 
                          quantity: float) -> Dict:
        """시장가 주문 생성"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            return order
        except BinanceAPIException as e:
            self.logger.error(f"시장가 주문 생성 실패: {e}")
            return {}
            
    def create_limit_order(self, 
                         symbol: str, 
                         side: str, 
                         quantity: float, 
                         price: float) -> Dict:
        """지정가 주문 생성"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price
            )
            return order
        except BinanceAPIException as e:
            self.logger.error(f"지정가 주문 생성 실패: {e}")
            return {}
            
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """주문 취소"""
        try:
            result = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            return result
        except BinanceAPIException as e:
            self.logger.error(f"주문 취소 실패: {e}")
            return {}
            
    def get_order(self, symbol: str, order_id: str) -> Dict:
        """주문 정보 조회"""
        try:
            order = self.client.get_order(
                symbol=symbol,
                orderId=order_id
            )
            return order
        except BinanceAPIException as e:
            self.logger.error(f"주문 정보 조회 실패: {e}")
            return {} 