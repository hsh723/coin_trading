"""
실행 전략 사용 예제

다양한 실행 전략을 사용하는 방법을 보여주는 예제입니다.
"""

import asyncio
import logging
from typing import Dict, Any

from src.execution.strategies import ExecutionStrategyFactory, AdaptiveExecutionStrategy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_adaptive_strategy_example():
    """적응형 전략 예제"""
    try:
        # 적응형 전략 설정
        config = {
            'slippage_threshold': 0.002,
            'volatility_threshold': 0.01,
            'max_participation_rate': 0.3,
            'initial_participation_rate': 0.1
        }
        
        # 전략 인스턴스 생성
        strategy = AdaptiveExecutionStrategy(config)
        
        # 주문 요청 준비
        order_request = {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'quantity': 0.1,
            'current_price': 60000.0,
            'urgency_factor': 0.7,  # 높은 긴급도
            'orderbook': {
                'asks': [[60010.0, 1.5], [60050.0, 2.0]],
                'bids': [[59990.0, 1.0], [59950.0, 1.8]]
            }
        }
        
        # 전략 실행
        logger.info(f"적응형 전략 실행 시작: {order_request['symbol']}")
        result = await strategy.execute(order_request)
        
        # 결과 출력
        logger.info(f"적응형 전략 실행 결과:")
        logger.info(f"  성공: {result['success']}")
        logger.info(f"  실행 수량: {result['executed_quantity']}")
        logger.info(f"  남은 수량: {result['remaining_quantity']}")
        if 'average_price' in result:
            logger.info(f"  평균 가격: {result['average_price']}")
        if 'error' in result:
            logger.info(f"  오류: {result['error']}")
            
    except Exception as e:
        logger.error(f"적응형 전략 예제 실행 중 오류 발생: {str(e)}")

async def run_factory_example():
    """전략 팩토리 사용 예제"""
    try:
        # 팩토리 설정
        factory_config = {
            'twap': {
                'time_window': 1800,  # 30분
                'slice_count': 6,
                'random_factor': 0.1
            },
            'vwap': {
                'time_window': 1800,  # 30분
                'interval_count': 6,
                'deviation_limit': 0.003
            },
            'adaptive': {
                'slippage_threshold': 0.002,
                'volatility_threshold': 0.01,
                'max_participation_rate': 0.3
            }
        }
        
        # 팩토리 생성
        factory = ExecutionStrategyFactory(factory_config)
        
        # 사용 가능한 전략 출력
        strategies = factory.get_available_strategies()
        logger.info(f"사용 가능한 전략: {strategies}")
        
        # TWAP 전략 가져오기
        twap_strategy = factory.get_strategy('twap')
        if twap_strategy:
            logger.info(f"TWAP 전략 인스턴스 생성 성공: {twap_strategy.get_name()}")
            
        # VWAP 전략 가져오기
        vwap_strategy = factory.get_strategy('vwap')
        if vwap_strategy:
            logger.info(f"VWAP 전략 인스턴스 생성 성공: {vwap_strategy.get_name()}")
            
        # 리소스 정리
        await factory.cleanup()
        
    except Exception as e:
        logger.error(f"전략 팩토리 예제 실행 중 오류 발생: {str(e)}")

async def main():
    """메인 함수"""
    logger.info("===== 실행 전략 예제 시작 =====")
    
    # 적응형 전략 예제 실행
    await run_adaptive_strategy_example()
    
    # 전략 팩토리 예제 실행
    await run_factory_example()
    
    logger.info("===== 실행 전략 예제 종료 =====")

if __name__ == "__main__":
    asyncio.run(main()) 