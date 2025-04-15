import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime, timedelta
import os
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.execution import TradeExecutor
from src.ml.bayesian.risk_management import RiskManager

class MockExchangeAPI:
    """거래소 API 모의 구현"""
    
    def __init__(self):
        self.orders = {}
        self.position = {'size': 0.0}
        self.portfolio_value = 10000.0
        self.current_price = 50000.0
        self.volatility = 0.02
    
    def place_order(self, symbol, side, type, amount, price=None):
        """주문 생성"""
        order_id = f"order_{len(self.orders)}"
        self.orders[order_id] = {
            'symbol': symbol,
            'side': side,
            'type': type,
            'amount': amount,
            'price': price,
            'status': 'filled',
            'execution_price': self.current_price,
            'execution_size': amount
        }
        return self.orders[order_id]
    
    def get_order_status(self, order_id):
        """주문 상태 조회"""
        return self.orders.get(order_id, {'status': 'unknown'})
    
    def cancel_order(self, order_id):
        """주문 취소"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'canceled'
    
    def get_position(self):
        """포지션 조회"""
        return self.position
    
    def get_portfolio_value(self):
        """포트폴리오 가치 조회"""
        return self.portfolio_value
    
    def get_current_price(self):
        """현재 가격 조회"""
        return self.current_price
    
    def get_volatility(self):
        """변동성 조회"""
        return self.volatility

async def main():
    """실행 예제"""
    try:
        # 거래소 API 초기화
        exchange_api = MockExchangeAPI()
        
        # 리스크 관리자 초기화
        risk_manager = RiskManager(
            initial_capital=10000.0,
            max_position_size=0.1,
            max_drawdown=0.2,
            var_confidence=0.95
        )
        
        # 거래 실행 시스템 초기화
        executor = TradeExecutor(
            exchange_api=exchange_api,
            risk_manager=risk_manager,
            max_position_size=0.1,
            max_slippage=0.001
        )
        
        # 거래 실행 시스템 시작
        await executor.start()
        
        # 샘플 주문 생성
        orders = [
            {
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'type': 'market',
                'size': 0.01,
                'creation_time': datetime.now()
            },
            {
                'symbol': 'BTC/USDT',
                'side': 'sell',
                'type': 'market',
                'size': 0.005,
                'creation_time': datetime.now()
            }
        ]
        
        # 주문 실행
        for order in orders:
            executor.place_order(order)
        
        # 잠시 대기
        await asyncio.sleep(5)
        
        # 실행 통계 조회
        stats = executor.get_execution_stats()
        logger.info(f"실행 통계: {stats}")
        
        # 거래 실행 시스템 중지
        await executor.stop()
        
    except Exception as e:
        logger.error(f"예제 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 