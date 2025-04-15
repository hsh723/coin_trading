"""
TWAP 실행 전략 모듈

시간 가중 평균 가격(Time-Weighted Average Price) 전략을 구현합니다.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from src.execution.strategies.base_strategy import BaseExecutionStrategy

logger = logging.getLogger(__name__)

class TwapExecutionStrategy(BaseExecutionStrategy):
    """TWAP 실행 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        TWAP 실행 전략 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        super().__init__(config)
        
        # 전략 특정 설정
        self.time_window = config.get('time_window', 3600)  # 실행 시간 간격(초)
        self.slice_count = config.get('slice_count', 10)  # 분할 횟수
        self.random_factor = config.get('random_factor', 0.2)  # 랜덤 변동 계수 (0.0 ~ 1.0)
        
        # 실행 상태
        self.is_active = False
        self.execution_task = None
        self.start_time = None
        self.end_time = None
        self.slice_interval = 0
        self.remaining_quantity = 0.0
        self.executed_quantity = 0.0
        self.order_side = ''
        self.symbol = ''
        self.execution_results = []
        
    async def execute(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        주문 실행
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # 주문 요청 검증
            await self._validate_request(order_request)
            
            # 실행 설정
            self.symbol = order_request['symbol']
            self.order_side = order_request['side']
            self.remaining_quantity = float(order_request['quantity'])
            self.executed_quantity = 0.0
            self.execution_results = []
            
            # 시간 창 설정 (요청에 제공된 경우 사용)
            if 'time_window' in order_request:
                self.time_window = float(order_request['time_window'])
                
            # 분할 횟수 설정 (요청에 제공된 경우 사용)
            if 'slice_count' in order_request:
                self.slice_count = int(order_request['slice_count'])
                
            # 시간 창이 너무 짧거나 분할 횟수가 너무 적은 경우 조정
            if self.time_window < 60:
                self.time_window = 60  # 최소 1분
                
            if self.slice_count < 2:
                self.slice_count = 2  # 최소 2회 분할
                
            # 시작/종료 시간 설정
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(seconds=self.time_window)
            
            # 구간 간격 계산
            self.slice_interval = self.time_window / self.slice_count
            
            # 활성화 상태로 설정
            self.is_active = True
            
            # 분할 실행 시작
            logger.info(
                f"TWAP 전략 실행 시작: {self.symbol}, 수량: {self.remaining_quantity}, "
                f"시간 창: {self.time_window}초, 분할 횟수: {self.slice_count}"
            )
            
            self.execution_task = asyncio.create_task(self._execute_slices())
            execution_result = await self.execution_task
            
            return execution_result
            
        except Exception as e:
            logger.error(f"TWAP 전략 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'twap',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'timestamp': datetime.now()
            }
            
    async def cancel(self) -> bool:
        """
        실행 취소
        
        Returns:
            bool: 취소 성공 여부
        """
        try:
            if not self.is_active:
                return True
                
            # 실행 중단
            self.is_active = False
            
            if self.execution_task and not self.execution_task.done():
                self.execution_task.cancel()
                
            logger.info(f"TWAP 전략 실행 취소: {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"TWAP 전략 취소 오류: {str(e)}")
            return False
            
    async def _validate_request(self, order_request: Dict[str, Any]) -> None:
        """
        주문 요청 검증
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
        """
        required_fields = ['symbol', 'side', 'quantity']
        for field in required_fields:
            if field not in order_request:
                raise ValueError(f"필수 필드 누락: {field}")
                
        # 수량 검증
        quantity = float(order_request['quantity'])
        if quantity <= 0:
            raise ValueError(f"유효하지 않은 수량: {quantity}")
            
        # 주문 유형 검증
        side = order_request['side'].upper()
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"유효하지 않은 주문 유형: {side}")
            
    async def _execute_slices(self) -> Dict[str, Any]:
        """
        분할 주문 실행
        
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # 각 조각의 기본 수량 계산
            base_slice_quantity = self.remaining_quantity / self.slice_count
            
            for i in range(self.slice_count):
                if not self.is_active or self.remaining_quantity <= 0:
                    break
                    
                # 다음 실행 시간 계산
                next_execution_time = self.start_time + timedelta(seconds=i * self.slice_interval)
                
                # 현재 시간이 다음 실행 시간보다 앞서면 대기
                wait_time = (next_execution_time - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
                # 랜덤 요소 적용 (±random_factor%)
                if self.random_factor > 0:
                    random_adjustment = np.random.uniform(
                        1 - self.random_factor,
                        1 + self.random_factor
                    )
                    adjusted_quantity = base_slice_quantity * random_adjustment
                else:
                    adjusted_quantity = base_slice_quantity
                    
                # 남은 수량 제한
                adjusted_quantity = min(adjusted_quantity, self.remaining_quantity)
                
                # 실행
                slice_result = await self._execute_slice(adjusted_quantity)
                self.execution_results.append(slice_result)
                
                if slice_result['success']:
                    executed_qty = float(slice_result.get('executed_qty', 0.0))
                    self.executed_quantity += executed_qty
                    self.remaining_quantity -= executed_qty
                    
                    logger.debug(
                        f"TWAP 조각 {i+1}/{self.slice_count} 실행 완료: {executed_qty:.6f}, "
                        f"남은 수량: {self.remaining_quantity:.6f}"
                    )
                else:
                    logger.warning(f"TWAP 조각 {i+1}/{self.slice_count} 실행 실패: {slice_result.get('error', 'unknown error')}")
                    
            # 남은 수량이 있고 마지막 실행이 성공적이었다면 남은 수량도 실행
            if self.is_active and self.remaining_quantity > 0:
                final_result = await self._execute_slice(self.remaining_quantity)
                self.execution_results.append(final_result)
                
                if final_result['success']:
                    executed_qty = float(final_result.get('executed_qty', 0.0))
                    self.executed_quantity += executed_qty
                    self.remaining_quantity -= executed_qty
                    
                    logger.debug(f"TWAP 최종 실행 완료: {executed_qty:.6f}")
                    
            # 실행 완료
            execution_complete = (self.remaining_quantity <= 0)
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            # 실행 결과 집계
            result = {
                'success': execution_complete,
                'strategy': 'twap',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'average_price': self._calculate_average_price(),
                'execution_time': execution_time,
                'slice_count': len(self.execution_results),
                'timestamp': datetime.now()
            }
            
            if not execution_complete:
                result['error'] = "실행 미완료"
                
            logger.info(
                f"TWAP 전략 실행 완료: 성공={execution_complete}, "
                f"실행량={self.executed_quantity:.6f}, 평균가={result['average_price']:.2f}"
            )
            
            return result
            
        except asyncio.CancelledError:
            logger.info("TWAP 전략 실행 취소됨")
            return {
                'success': False,
                'error': 'cancelled',
                'strategy': 'twap',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"TWAP 전략 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'twap',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'timestamp': datetime.now()
            }
        finally:
            self.is_active = False
            
    async def _execute_slice(self, quantity: float) -> Dict[str, Any]:
        """
        주문 조각 실행
        
        Args:
            quantity (float): 실행 수량
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # TODO: 실제 구현에서는 실제 거래 실행 로직
            
            # 성공적인 실행으로 가정한 샘플 응답
            current_price = 100.0  # 샘플 가격
            
            return {
                'success': True,
                'executed_qty': quantity,
                'price': current_price,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"조각 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    def _calculate_average_price(self) -> float:
        """
        평균 실행 가격 계산
        
        Returns:
            float: 평균 실행 가격
        """
        total_qty = 0.0
        total_value = 0.0
        
        for result in self.execution_results:
            if result.get('success', False):
                qty = float(result.get('executed_qty', 0.0))
                price = float(result.get('price', 0.0))
                
                if qty > 0 and price > 0:
                    total_qty += qty
                    total_value += qty * price
                    
        if total_qty > 0:
            return total_value / total_qty
        return 0.0 