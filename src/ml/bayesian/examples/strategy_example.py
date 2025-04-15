import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from ..trading_strategy import TradingStrategy

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockDataGenerator:
    """시뮬레이션을 위한 데이터 생성기"""
    
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.01):
        self.price = initial_price
        self.volatility = volatility
        self.trend = 0.001  # 약한 상승 추세
        
    def generate_data(self) -> Dict[str, Any]:
        """새로운 시장 데이터 생성"""
        # 가격 변화 생성
        price_change = np.random.normal(self.trend, self.volatility)
        self.price *= (1 + price_change)
        
        # 타임스탬프 생성
        timestamp = datetime.now()
        
        return {
            'price': self.price,
            'timestamp': timestamp
        }

async def main():
    """메인 실행 함수"""
    try:
        # 전략 초기화
        strategy = TradingStrategy(
            lookback_period=20,
            rsi_threshold=30.0,
            volatility_threshold=0.02,
            min_confidence=0.7
        )
        
        # 데이터 생성기 초기화
        data_generator = MockDataGenerator()
        
        # 시뮬레이션 실행
        logger.info("트레이딩 전략 시뮬레이션 시작")
        
        for _ in range(100):
            # 새로운 데이터 생성
            data = data_generator.generate_data()
            
            # 신호 생성
            signal = strategy.get_signal(data)
            
            # 결과 출력
            if signal != 0:
                action = "매수" if signal == 1 else "매도"
                logger.info(f"시간: {data['timestamp']}, 가격: {data['price']:.2f}, 신호: {action}, 신뢰도: {strategy.signal_confidence:.2f}")
            
            # 대기
            await asyncio.sleep(1)
        
        logger.info("시뮬레이션 종료")
        
    except Exception as e:
        logger.error(f"시뮬레이션 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 