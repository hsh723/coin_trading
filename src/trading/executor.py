from typing import Dict, Optional
import logging
from datetime import datetime
from src.exchange.binance_api import BinanceAPI

class OrderExecutor:
    """주문 실행 클래스"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: API 키
            api_secret: API 시크릿
        """
        self.exchange = BinanceAPI(api_key, api_secret)
        self.logger = logging.getLogger(__name__)
        
    async def execute_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Dict:
        """
        주문 실행
        
        Args:
            symbol: 거래 쌍
            side: 매수/매도
            quantity: 수량
            price: 가격 (지정가 주문 시)
            
        Returns:
            Dict: 주문 결과
        """
        try:
            if price:
                # 지정가 주문
                order = await self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price
                )
            else:
                # 시장가 주문
                order = await self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity
                )
                
            self.logger.info(f"주문 실행 성공: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"주문 실행 실패: {str(e)}")
            raise
            
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        주문 취소
        
        Args:
            symbol: 거래 쌍
            order_id: 주문 ID
            
        Returns:
            Dict: 취소 결과
        """
        try:
            result = await self.exchange.cancel_order(symbol=symbol, order_id=order_id)
            self.logger.info(f"주문 취소 성공: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {str(e)}")
            raise
            
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """
        주문 상태 조회
        
        Args:
            symbol: 거래 쌍
            order_id: 주문 ID
            
        Returns:
            Dict: 주문 상태
        """
        try:
            status = await self.exchange.get_order(symbol=symbol, order_id=order_id)
            return status
            
        except Exception as e:
            self.logger.error(f"주문 상태 조회 실패: {str(e)}")
            raise 