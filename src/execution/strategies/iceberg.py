"""
Iceberg 실행 전략
"""

from datetime import datetime
from typing import Dict, Any
from .base import ExecutionStrategy
import time

class IcebergStrategy(ExecutionStrategy):
    def __init__(self, config: Dict[str, Any]):
        """
        Iceberg 전략 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        super().__init__(config)
        self.active_orders = {}
        self.max_slice_size = config.get('max_slice_size', 0.1)  # 기본값 10%
        self.min_slice_size = config.get('min_slice_size', 0.05)  # 기본값 5%
        self.slice_interval = config.get('slice_interval', 60)  # 기본값 60초
        self.display_size = config.get('display_size', 0.1)  # 기본값 10%
        self.refresh_interval = config.get('refresh_interval', 60)  # 기본값 60초
        
    def execute(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Iceberg 전략으로 주문 실행
        
        Args:
            order (Dict[str, Any]): 주문 정보
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # 주문 파라미터 검증
            if not all(k in order for k in ['symbol', 'side', 'quantity', 'price']):
                raise ValueError("필수 주문 파라미터가 누락되었습니다")
                
            # 슬라이스 크기 계산
            slice_size = self._calculate_slice_size(order)
            
            # 주문 실행 결과
            result = {
                'order_id': f"ICEBERG_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'status': 'ACTIVE',
                'executed_quantity': 0.0,
                'remaining_quantity': order['quantity'],
                'slices': []
            }
            
            # 슬라이스 실행
            while result['remaining_quantity'] > 0:
                current_slice = min(slice_size, result['remaining_quantity'])
                slice_result = self._execute_slice(order, current_slice)
                result['slices'].append(slice_result)
                result['executed_quantity'] += slice_result['executed_quantity']
                result['remaining_quantity'] -= slice_result['executed_quantity']
                
                # 슬라이스 간격 대기
                time.sleep(self.slice_interval)
                
            return result
            
        except Exception as e:
            self.logger.error(f"아이스버그 전략 실행 중 오류 발생: {str(e)}")
            raise
            
    async def cancel(self, order_id: str) -> Dict[str, Any]:
        """
        주문 취소
        
        Args:
            order_id (str): 주문 ID
            
        Returns:
            Dict[str, Any]: 취소 결과
        """
        try:
            if order_id in self.active_orders:
                order = self.active_orders.pop(order_id)
                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': order['symbol'],
                    'status': 'cancelled',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'주문을 찾을 수 없음: {order_id}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _calculate_slice_size(self, order_params: Dict[str, Any]) -> float:
        """슬라이스 크기 계산"""
        try:
            # 최소/최대 슬라이스 크기 계산
            min_size = order_params['quantity'] * self.min_slice_size
            max_size = order_params['quantity'] * self.max_slice_size
            
            # 시장 상황에 따른 동적 조정
            market_volume = self._get_market_volume(order_params['symbol'])
            if market_volume > 0:
                # 시장 거래량의 1%를 기준으로 조정
                target_size = market_volume * 0.01
                slice_size = min(max_size, max(min_size, target_size))
            else:
                slice_size = max_size
                
            return slice_size
            
        except Exception as e:
            self.logger.error(f"슬라이스 크기 계산 중 오류 발생: {str(e)}")
            return order_params['quantity'] * self.max_slice_size
            
    def _execute_slice(self, order_params: Dict[str, Any], slice_size: float) -> Dict[str, Any]:
        """슬라이스 주문 실행"""
        try:
            # 슬라이스 주문 파라미터 생성
            slice_params = {
                'symbol': order_params['symbol'],
                'side': order_params['side'],
                'quantity': slice_size,
                'price': order_params['price'],
                'type': 'LIMIT'
            }
            
            # 주문 실행
            order_result = self.exchange.create_order(**slice_params)
            
            # 결과 처리
            return {
                'order_id': order_result['orderId'],
                'executed_quantity': float(order_result['executedQty']),
                'price': float(order_result['price']),
                'status': order_result['status'],
                'timestamp': order_result['time']
            }
            
        except Exception as e:
            self.logger.error(f"슬라이스 주문 실행 중 오류 발생: {str(e)}")
            raise
            
    def _get_market_volume(self, symbol: str) -> float:
        """시장 거래량 조회"""
        try:
            ticker = self.exchange.get_ticker(symbol=symbol)
            return float(ticker['volume'])
        except Exception as e:
            self.logger.error(f"시장 거래량 조회 중 오류 발생: {str(e)}")
            return 0.0 