"""
트레이딩 시스템 모듈

이 모듈은 전체 트레이딩 시스템을 관리하는 역할을 담당합니다.
주요 기능:
- 시스템 초기화 및 종료
- 주문 실행 관리
- 리스크 관리
- 성능 모니터링
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.execution.execution_manager import ExecutionManager
from src.risk.risk_manager import RiskManager
from src.monitoring.performance_metrics import PerformanceMetricsCollector
from src.position.position_manager import PositionManager

logger = logging.getLogger(__name__)

class TradingSystem:
    """트레이딩 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        트레이딩 시스템 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 리스크 매니저 초기화
        risk_config = {
            'initial_capital': config.get('initial_capital', 100000.0),
            'position_limits': {
                'max_position_size': config.get('max_position_size', 0.1),
                'max_order_size': config.get('max_order_size', 0.05)
            },
            'risk_limits': {
                'max_drawdown': config.get('max_drawdown', 0.2),
                'stop_loss': config.get('stop_loss', 0.05),
                'take_profit': config.get('take_profit', 0.1),
                'trailing_stop': config.get('trailing_stop', 0.03),
                'max_concentration': config.get('max_concentration', 0.3)
            },
            'volatility_limits': {
                'threshold': config.get('volatility_threshold', 0.02)
            }
        }
        self.risk_manager = RiskManager(risk_config)
        
        # 실행 매니저 초기화
        self.execution_manager = ExecutionManager(config)
        
        # 포지션 매니저 초기화
        self.position_manager = PositionManager(config)
        
        # 성능 메트릭 수집기 초기화
        self.performance_metrics = PerformanceMetricsCollector(config)
        
        # 거래 이력 초기화
        self.trade_history = []
        
        # 시스템 상태 초기화
        self.is_initialized = False
        self.is_running = False
        
        # 상태 관리
        self.active_orders = {}
        self.order_history = []
        
    async def initialize(self):
        """트레이딩 시스템 초기화"""
        try:
            # 컴포넌트 초기화
            await self.execution_manager.initialize()
            await self.risk_manager.initialize()
            await self.performance_metrics.initialize()
            
            # 시스템 상태 업데이트
            self.is_initialized = True
            self.is_running = True
            
            self.logger.info("트레이딩 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"트레이딩 시스템 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """시스템 종료"""
        try:
            self.is_running = False
            
            # 컴포넌트 종료
            await self.execution_manager.close()
            await self.risk_manager.close()
            await self.performance_metrics.close()
            
            self.logger.info("트레이딩 시스템 종료 완료")
            
        except Exception as e:
            self.logger.error(f"트레이딩 시스템 종료 실패: {str(e)}")
            raise
            
    async def execute_trade(self, order: Dict[str, Any], position: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        거래 실행
        
        Args:
            order (Dict[str, Any]): 주문 정보
            position (Dict[str, Any], optional): 포지션 정보
            
        Returns:
            Dict[str, Any]: 거래 실행 결과
        """
        try:
            # 시스템 초기화 확인
            if not self.is_initialized:
                return {
                    'success': False,
                    'error': 'system_not_initialized'
                }
                
            # 포지션 정보 조회
            if position is None:
                position_result = await self.position_manager.get_position(order['symbol'])
                position = position_result.get('position', {
                    'symbol': order['symbol'],
                    'size': 0.0,
                    'entry_price': order['price'],
                    'unrealized_pnl': 0.0,
                    'side': order['side']
                })
            
            # 리스크 체크
            risk_check = await self.risk_manager.check_risk(order, position)
            if risk_check['is_risky']:
                return {
                    'success': False,
                    'error': 'risk_limit_exceeded',
                    'details': risk_check
                }
                
            # 주문 실행을 위한 주문 정보 복사
            execution_order = order.copy()
            execution_order['quantity'] = execution_order['size']
            
            # 주문 실행
            execution_result = await self.execution_manager.execute_order(execution_order)
            if not execution_result['success']:
                return execution_result
                
            # 활성 주문 추가
            self.active_orders[execution_result['order_id']] = {
                'order': order,
                'execution': execution_result,
                'timestamp': datetime.now()
            }
            
            # 포지션 업데이트
            new_position = {
                'symbol': order['symbol'],
                'size': position['size'] + order['size'],
                'entry_price': order['price'],
                'unrealized_pnl': 0.0,
                'side': order['side']
            }
            await self.position_manager.update_position(order['symbol'], new_position)
            
            # 거래 이력 추가
            self.trade_history.append({
                'timestamp': datetime.now(),
                'order': order,
                'execution': execution_result
            })
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류 발생: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        try:
            if order_id not in self.active_orders:
                return {
                    'success': False,
                    'error': f"주문을 찾을 수 없음: {order_id}"
                }
                
            # 주문 취소
            cancellation_result = await self.execution_manager.cancel_order(order_id)
            
            if cancellation_result['success']:
                # 성공적인 취소 처리
                order_info = self.active_orders.pop(order_id)
                self.order_history.append({
                    'order': order_info['order'],
                    'execution': order_info['execution'],
                    'cancellation': cancellation_result,
                    'timestamp': datetime.now()
                })
                
            return cancellation_result
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """리스크 메트릭 조회"""
        try:
            return await self.risk_manager.get_risk_metrics()
            
        except Exception as e:
            self.logger.error(f"리스크 메트릭 조회 실패: {str(e)}")
            return {}
            
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        try:
            return await self.performance_metrics.get_metrics()
            
        except Exception as e:
            self.logger.error(f"성능 메트릭 조회 실패: {str(e)}")
            return {}
            
    def get_active_orders(self) -> Dict[str, Any]:
        """활성 주문 조회"""
        return self.active_orders.copy()
        
    def get_order_history(self) -> List[Dict[str, Any]]:
        """주문 이력 조회"""
        return self.order_history.copy()
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """거래 이력 조회"""
        return self.trade_history.copy()

    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        포지션 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Dict[str, Any]: 포지션 정보
        """
        try:
            # 테스트용 더미 구현
            return {
                'symbol': symbol,
                'size': 0.0,
                'entry_price': 0.0,
                'side': 'long'
            }
            
        except Exception as e:
            self.logger.error(f"포지션 조회 실패: {str(e)}")
            return {
                'symbol': symbol,
                'size': 0.0,
                'entry_price': 0.0,
                'side': 'long'
            } 