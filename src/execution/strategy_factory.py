"""
실행 전략 팩토리 모듈

이 모듈은 다양한 실행 전략을 생성하고 관리하는 역할을 담당합니다.
주요 기능:
- TWAP, VWAP, Iceberg 등 다양한 실행 전략 생성
- 전략별 설정 관리
- 전략 최적화 지원
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ExecutionStrategy(ABC):
    """실행 전략 추상 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.is_active = False
        
    @abstractmethod
    async def execute(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """주문 실행"""
        pass
        
    @abstractmethod
    async def cancel(self, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        pass
        
    async def initialize(self):
        """전략 초기화"""
        try:
            self.is_active = True
            logger.info(f"{self.name} 전략 초기화 완료")
        except Exception as e:
            logger.error(f"{self.name} 전략 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """전략 종료"""
        try:
            self.is_active = False
            logger.info(f"{self.name} 전략 종료 완료")
        except Exception as e:
            logger.error(f"{self.name} 전략 종료 실패: {str(e)}")
            raise

class TWAPStrategy(ExecutionStrategy):
    """TWAP (Time-Weighted Average Price) 실행 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = 'twap'
        self.interval = config.get('interval', 60)  # 초
        self.num_slices = config.get('num_slices', 10)
        
    async def execute(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """TWAP 전략으로 주문 실행"""
        try:
            if not self.is_active:
                raise RuntimeError("전략이 활성화되지 않았습니다.")
                
            # 주문 정보 추출
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']
            price = order.get('price', None)
            
            # 슬라이스 크기 계산
            slice_quantity = quantity / self.num_slices
            
            # 실행 결과 초기화
            results = []
            total_executed = 0
            
            # 슬라이스별 실행
            for i in range(self.num_slices):
                # 슬라이스 주문 생성
                slice_order = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': slice_quantity,
                    'price': price
                }
                
                # 슬라이스 실행
                result = await self._execute_slice(slice_order)
                results.append(result)
                
                if result.get('success', False):
                    total_executed += result.get('executed_quantity', 0)
                    
                # 다음 슬라이스까지 대기
                await asyncio.sleep(self.interval)
                
            # 전체 실행 결과 반환
            return {
                'success': total_executed > 0,
                'order_id': order.get('order_id', f"twap_{datetime.now().timestamp()}"),
                'symbol': symbol,
                'side': side,
                'total_quantity': quantity,
                'executed_quantity': total_executed,
                'results': results,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"TWAP 실행 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def cancel(self, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        try:
            # TWAP 전략에서는 모든 슬라이스 주문을 취소
            return {
                'success': True,
                'order_id': order_id,
                'status': 'cancelled',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"TWAP 취소 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def _execute_slice(self, slice_order: Dict[str, Any]) -> Dict[str, Any]:
        """슬라이스 주문 실행"""
        # 실제 거래소 API 호출 대신 테스트용 더미 구현
        return {
            'success': True,
            'order_id': f"slice_{datetime.now().timestamp()}",
            'symbol': slice_order['symbol'],
            'side': slice_order['side'],
            'quantity': slice_order['quantity'],
            'executed_quantity': slice_order['quantity'],
            'price': slice_order.get('price', 50000.0),
            'timestamp': datetime.now()
        }

class VWAPStrategy(ExecutionStrategy):
    """VWAP (Volume-Weighted Average Price) 실행 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = 'vwap'
        self.window_size = config.get('window_size', 100)
        self.volume_profile = config.get('volume_profile', 'historical')
        
    async def execute(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """VWAP 전략으로 주문 실행"""
        try:
            if not self.is_active:
                raise RuntimeError("전략이 활성화되지 않았습니다.")
                
            # 주문 정보 추출
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']
            price = order.get('price', None)
            
            # 볼륨 프로파일 분석
            volume_profile = await self._analyze_volume_profile(symbol)
            
            # 실행 계획 수립
            execution_plan = self._create_execution_plan(quantity, volume_profile)
            
            # 실행 결과 초기화
            results = []
            total_executed = 0
            
            # 계획에 따라 실행
            for step in execution_plan:
                # 단계별 주문 생성
                step_order = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': step['quantity'],
                    'price': price
                }
                
                # 단계 실행
                result = await self._execute_step(step_order)
                results.append(result)
                
                if result.get('success', False):
                    total_executed += result.get('executed_quantity', 0)
                    
                # 다음 단계까지 대기
                await asyncio.sleep(step['delay'])
                
            # 전체 실행 결과 반환
            return {
                'success': total_executed > 0,
                'order_id': order.get('order_id', f"vwap_{datetime.now().timestamp()}"),
                'symbol': symbol,
                'side': side,
                'total_quantity': quantity,
                'executed_quantity': total_executed,
                'results': results,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"VWAP 실행 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def cancel(self, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        try:
            # VWAP 전략에서는 모든 단계 주문을 취소
            return {
                'success': True,
                'order_id': order_id,
                'status': 'cancelled',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"VWAP 취소 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def _analyze_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """볼륨 프로파일 분석"""
        # 실제 거래소 API 호출 대신 테스트용 더미 구현
        return {
            'total_volume': 1000.0,
            'price_levels': [
                {'price': 50000.0, 'volume': 200.0},
                {'price': 50100.0, 'volume': 300.0},
                {'price': 50200.0, 'volume': 500.0}
            ]
        }
        
    def _create_execution_plan(self, quantity: float, volume_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """실행 계획 수립"""
        total_volume = volume_profile['total_volume']
        execution_plan = []
        
        for level in volume_profile['price_levels']:
            volume_ratio = level['volume'] / total_volume
            step_quantity = quantity * volume_ratio
            execution_plan.append({
                'quantity': step_quantity,
                'delay': 1.0  # 초
            })
            
        return execution_plan
        
    async def _execute_step(self, step_order: Dict[str, Any]) -> Dict[str, Any]:
        """단계별 주문 실행"""
        # 실제 거래소 API 호출 대신 테스트용 더미 구현
        return {
            'success': True,
            'order_id': f"step_{datetime.now().timestamp()}",
            'symbol': step_order['symbol'],
            'side': step_order['side'],
            'quantity': step_order['quantity'],
            'executed_quantity': step_order['quantity'],
            'price': step_order.get('price', 50000.0),
            'timestamp': datetime.now()
        }

class IcebergStrategy(ExecutionStrategy):
    """Iceberg 실행 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = 'iceberg'
        self.visible_size = config.get('visible_size', 0.1)  # 전체 수량의 비율
        self.slice_interval = config.get('slice_interval', 60)  # 초
        
    async def execute(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Iceberg 전략으로 주문 실행"""
        try:
            if not self.is_active:
                raise RuntimeError("전략이 활성화되지 않았습니다.")
                
            # 주문 정보 추출
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']
            price = order.get('price', None)
            
            # 가시적 크기 계산
            visible_quantity = quantity * self.visible_size
            hidden_quantity = quantity - visible_quantity
            
            # 가시적 주문 실행
            visible_result = await self._execute_visible_order({
                'symbol': symbol,
                'side': side,
                'quantity': visible_quantity,
                'price': price
            })
            
            if not visible_result.get('success', False):
                return visible_result
                
            # 숨겨진 주문 실행
            hidden_results = []
            total_executed = visible_result.get('executed_quantity', 0)
            
            while total_executed < quantity:
                # 숨겨진 슬라이스 주문 생성
                remaining = quantity - total_executed
                slice_quantity = min(visible_quantity, remaining)
                
                slice_order = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': slice_quantity,
                    'price': price
                }
                
                # 슬라이스 실행
                result = await self._execute_hidden_slice(slice_order)
                hidden_results.append(result)
                
                if result.get('success', False):
                    total_executed += result.get('executed_quantity', 0)
                    
                # 다음 슬라이스까지 대기
                await asyncio.sleep(self.slice_interval)
                
            # 전체 실행 결과 반환
            return {
                'success': total_executed > 0,
                'order_id': order.get('order_id', f"iceberg_{datetime.now().timestamp()}"),
                'symbol': symbol,
                'side': side,
                'total_quantity': quantity,
                'executed_quantity': total_executed,
                'visible_result': visible_result,
                'hidden_results': hidden_results,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Iceberg 실행 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def cancel(self, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        try:
            # Iceberg 전략에서는 가시적 주문과 숨겨진 주문을 모두 취소
            return {
                'success': True,
                'order_id': order_id,
                'status': 'cancelled',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Iceberg 취소 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def _execute_visible_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """가시적 주문 실행"""
        # 실제 거래소 API 호출 대신 테스트용 더미 구현
        return {
            'success': True,
            'order_id': f"visible_{datetime.now().timestamp()}",
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity'],
            'executed_quantity': order['quantity'],
            'price': order.get('price', 50000.0),
            'timestamp': datetime.now()
        }
        
    async def _execute_hidden_slice(self, slice_order: Dict[str, Any]) -> Dict[str, Any]:
        """숨겨진 슬라이스 주문 실행"""
        # 실제 거래소 API 호출 대신 테스트용 더미 구현
        return {
            'success': True,
            'order_id': f"hidden_{datetime.now().timestamp()}",
            'symbol': slice_order['symbol'],
            'side': slice_order['side'],
            'quantity': slice_order['quantity'],
            'executed_quantity': slice_order['quantity'],
            'price': slice_order.get('price', 50000.0),
            'timestamp': datetime.now()
        }

class ExecutionStrategyFactory:
    """실행 전략 팩토리"""
    
    def __init__(self):
        self.strategies = {}
        self.default_strategy = 'twap'
        
    async def initialize(self):
        """팩토리 초기화"""
        try:
            # 기본 전략 등록
            self.register_strategy('twap', TWAPStrategy)
            self.register_strategy('vwap', VWAPStrategy)
            self.register_strategy('iceberg', IcebergStrategy)
            
            logger.info("실행 전략 팩토리 초기화 완료")
            
        except Exception as e:
            logger.error(f"실행 전략 팩토리 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """팩토리 종료"""
        try:
            # 등록된 모든 전략 종료
            for strategy in self.strategies.values():
                if strategy.is_active:
                    await strategy.close()
                    
            logger.info("실행 전략 팩토리 종료 완료")
            
        except Exception as e:
            logger.error(f"실행 전략 팩토리 종료 실패: {str(e)}")
            raise
            
    def register_strategy(self, name: str, strategy_class: type):
        """전략 등록"""
        if name in self.strategies:
            raise ValueError(f"이미 등록된 전략: {name}")
            
        self.strategies[name] = strategy_class
        logger.info(f"전략 등록 완료: {name}")
        
    def create_strategy(self, name: str, config: Dict[str, Any]) -> ExecutionStrategy:
        """전략 생성"""
        if name not in self.strategies:
            raise ValueError(f"등록되지 않은 전략: {name}")
            
        strategy_class = self.strategies[name]
        return strategy_class(config)
        
    def get_available_strategies(self) -> List[str]:
        """사용 가능한 전략 목록 조회"""
        return list(self.strategies.keys())
        
    def set_default_strategy(self, name: str):
        """기본 전략 설정"""
        if name not in self.strategies:
            raise ValueError(f"등록되지 않은 전략: {name}")
            
        self.default_strategy = name
        logger.info(f"기본 전략 설정: {name}")
        
    def get_default_strategy(self) -> str:
        """기본 전략 조회"""
        return self.default_strategy 