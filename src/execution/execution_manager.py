"""
실행 매니저 모듈

이 모듈은 거래 실행 시스템의 전체적인 관리를 담당합니다.
주요 기능:
- 다양한 실행 전략 관리
- 시장 상태 기반 전략 선택
- 실행 모니터링 및 최적화
- 실행 성능 메트릭 관리
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import numpy as np
from collections import deque
import time
import os

from src.execution.market_state_monitor import MarketStateMonitor
from src.execution.execution_monitor import ExecutionMonitor
from src.execution.execution_quality_monitor import ExecutionQualityMonitor
from src.execution.error_handler import ErrorHandler
from src.execution.notifier import ExecutionNotifier
from src.execution.logger import ExecutionLogger
from src.execution.asset_cache_manager import AssetCacheManager
from src.execution.performance_metrics import PerformanceMetricsCollector
from src.execution.strategy_optimizer import ExecutionStrategyOptimizer
from src.execution.strategies import ExecutionStrategyFactory, TWAPStrategy, VWAPStrategy, IcebergStrategy
from src.exchange.binance_client import BinanceClient
from src.config.env_loader import EnvLoader
from src.execution.position_manager import PositionManager

logger = logging.getLogger(__name__)

class ExecutionManager:
    """실행 관리자"""
    def __init__(self, config: dict = None, test_mode: bool = False):
        """초기화"""
        self.config = config or {}
        self.test_mode = self.config.get('test_mode', test_mode)
        self.env_loader = EnvLoader()
        self.exchange_client = None
        self.active_executions = {}
        
        # 모니터 및 매니저 초기화
        market_monitor_config = self.config.get('market_monitor', {})
        self.market_monitor = MarketStateMonitor(market_monitor_config)
        
        execution_monitor_config = self.config.get('execution_monitor', {})
        self.execution_monitor = ExecutionMonitor(execution_monitor_config)
        
        quality_monitor_config = self.config.get('quality_monitor', {})
        self.quality_monitor = ExecutionQualityMonitor(quality_monitor_config)
        
        error_handler_config = self.config.get('error_handler', {})
        self.error_handler = ErrorHandler(error_handler_config)
        
        notifier_config = self.config.get('notifier', {})
        self.notifier = ExecutionNotifier(notifier_config)
        
        asset_cache_config = self.config.get('asset_cache', {})
        self.asset_cache = AssetCacheManager(asset_cache_config)
        
        metrics_config = self.config.get('metrics', {})
        self.performance_metrics = PerformanceMetricsCollector(metrics_config)
        
        optimizer_config = self.config.get('optimizer', {})
        self.strategy_optimizer = ExecutionStrategyOptimizer(optimizer_config)
        
        # 포지션 관리자 초기화
        position_manager_config = self.config.get('position_manager', {})
        self.position_manager = PositionManager(position_manager_config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("실행 관리자 인스턴스 생성")

    async def initialize(self) -> bool:
        """실행 관리자 초기화"""
        try:
            # API 키 설정
            if self.config.get('test_mode', False):
                api_key = 'test_key'
                api_secret = 'test_secret'
            else:
                api_key = self.config.get('api_key')
                api_secret = self.config.get('api_secret')
                
            if not api_key or not api_secret:
                self.logger.error("바이낸스 API 키가 설정되지 않았습니다")
                return False
                
            # 바이낸스 클라이언트 초기화
            self.exchange_client = BinanceClient(
                api_key=api_key,
                api_secret=api_secret,
                test_mode=self.config.get('test_mode', False)
            )
            await self.exchange_client.initialize()
            
            # 모니터 초기화
            self.market_monitor = MarketStateMonitor(self.config.get('market_monitor', {}))
            self.execution_monitor = ExecutionMonitor(self.config.get('execution_monitor', {}))
            self.quality_monitor = ExecutionQualityMonitor(self.config.get('quality_monitor', {}))
            
            # 기타 컴포넌트 초기화
            self.error_handler = ErrorHandler(self.config.get('error_handler', {}))
            self.notifier = ExecutionNotifier(self.config.get('notifier', {}))
            self.asset_cache = AssetCacheManager(self.config.get('asset_cache', {}))
            self.performance_metrics = PerformanceMetricsCollector(self.config.get('performance_metrics', {}))
            self.strategy_optimizer = ExecutionStrategyOptimizer(self.config.get('strategy_optimizer', {}))
            
            # 포지션 관리자 초기화
            if not await self.position_manager.initialize():
                self.logger.error("포지션 관리자 초기화 실패")
                return False
                
            self.logger.info("실행 관리자가 성공적으로 초기화되었습니다")
            return True
            
        except Exception as e:
            self.logger.error(f"실행 관리자 초기화 중 오류 발생: {str(e)}")
            return False

    async def execute_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """주문 실행"""
        try:
            if not self.exchange_client:
                raise RuntimeError("거래소 클라이언트가 초기화되지 않았습니다.")
                
            # 주문 실행
            order = await self.exchange_client.create_order(**order_params)
            
            # 실행 정보 저장
            execution_id = order["orderId"]
            self.active_executions[execution_id] = {
                "order": order,
                "status": "NEW",
                "timestamp": order.get("transactTime", None)
            }
            
            self.logger.info(f"주문 실행 성공: {execution_id}")
            return order
            
        except Exception as e:
            self.logger.error(f"주문 실행 실패: {str(e)}")
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """주문 취소"""
        try:
            if not self.exchange_client:
                raise RuntimeError("거래소 클라이언트가 초기화되지 않았습니다.")
                
            # 주문 취소
            result = await self.exchange_client.cancel_order(symbol, order_id)
            
            # 실행 정보 업데이트
            if order_id in self.active_executions:
                self.active_executions[order_id]["status"] = "CANCELED"
            
            self.logger.info(f"주문 취소 성공: {order_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {str(e)}")
            return False

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """시장 데이터 조회"""
        try:
            if not self.exchange_client:
                raise RuntimeError("거래소 클라이언트가 초기화되지 않았습니다.")
                
            return await self.exchange_client.get_market_data(symbol)
            
        except Exception as e:
            self.logger.error(f"시장 데이터 조회 실패: {str(e)}")
            raise

    async def close(self):
        """종료"""
        self.logger.info("실행 관리자 종료 중...")
        self.active_executions.clear()
        self.logger.info("실행 관리자 종료 완료")

    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        포지션 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Dict[str, Any]: 포지션 정보
        """
        try:
            if not self.position_manager:
                raise RuntimeError("포지션 관리자가 초기화되지 않았습니다.")
                
            return await self.position_manager.get_position(symbol)
            
        except Exception as e:
            logger.error(f"포지션 조회 실패: {str(e)}")
            raise

    async def adjust_position(self, symbol: str, target_size: float) -> Dict[str, Any]:
        """
        포지션 조정
        
        Args:
            symbol (str): 거래 심볼
            target_size (float): 목표 포지션 크기
            
        Returns:
            Dict[str, Any]: 조정 결과
        """
        try:
            if not self.position_manager:
                raise RuntimeError("포지션 관리자가 초기화되지 않았습니다.")
                
            return await self.position_manager.adjust_position(symbol, target_size)
            
        except Exception as e:
            logger.error(f"포지션 조정 실패: {str(e)}")
            raise

    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        호가 정보 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            Dict[str, Any]: 호가 정보
        """
        try:
            if not self.market_monitor:
                raise RuntimeError("시장 모니터가 초기화되지 않았습니다.")
                
            return await self.market_monitor.get_order_book(symbol)
            
        except Exception as e:
            logger.error(f"호가 정보 조회 실패: {str(e)}")
            raise

    async def check_risk(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """위험 검사"""
        try:
            if not self.exchange_client:
                raise RuntimeError("거래소 클라이언트가 초기화되지 않았습니다.")
                
            # 위험 노출 계산
            risk_exposure = await self._calculate_risk_exposure(order)
            
            # 레버리지 계산
            leverage = self.config.get('max_leverage', 1.0)
            current_leverage = risk_exposure / self.config.get('margin_balance', 1.0)
            
            # 위험 수준 계산
            risk_level = 'LOW'
            if current_leverage > leverage * 0.8:
                risk_level = 'HIGH'
            elif current_leverage > leverage * 0.5:
                risk_level = 'MEDIUM'
            
            # 위험 한도 체크
            max_exposure = self.config.get('max_position_size', 1.0)
            if risk_exposure > max_exposure or current_leverage > leverage:
                return {
                    'exposure': risk_exposure,
                    'leverage': current_leverage,
                    'limit': max_exposure,
                    'max_leverage': leverage,
                    'risk_level': 'CRITICAL',
                    'status': 'EXCEEDED',
                    'message': f'위험 노출이 한도를 초과했습니다: 노출={risk_exposure}, 레버리지={current_leverage}'
                }
                
            return {
                'exposure': risk_exposure,
                'leverage': current_leverage,
                'limit': max_exposure,
                'max_leverage': leverage,
                'risk_level': risk_level,
                'status': 'ACCEPTABLE',
                'message': '위험 수준이 허용 범위 내입니다'
            }
            
        except Exception as e:
            self.logger.error(f"위험 검사 실패: {str(e)}")
            return {
                'exposure': 0.0,
                'leverage': 0.0,
                'limit': 0.0,
                'max_leverage': 0.0,
                'risk_level': 'UNKNOWN',
                'status': 'ERROR',
                'message': str(e)
            }

    async def set_risk_limits(self, limits: Dict[str, float]) -> Dict[str, Any]:
        """
        위험 한도 설정
        
        Args:
            limits (Dict[str, float]): 위험 한도 설정
            
        Returns:
            Dict[str, Any]: 설정 결과
        """
        try:
            # 위험 한도 검증
            for key, value in limits.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"잘못된 위험 한도 값: {key}={value}")
                    
            # 위험 한도 업데이트
            self.config.update(limits)
            
            return {
                'success': True,
                'limits': limits
            }
            
        except Exception as e:
            logger.error(f"위험 한도 설정 실패: {str(e)}")
            raise

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        try:
            if not self.performance_metrics:
                raise RuntimeError("성능 메트릭 수집기가 초기화되지 않았습니다.")
                
            return await self.performance_metrics.get_metrics()
            
        except Exception as e:
            self.logger.error(f"성능 메트릭 조회 실패: {str(e)}")
            raise

    async def measure_latency(self, operation: str = 'order') -> Dict[str, Any]:
        """
        지연시간 측정
        
        Args:
            operation (str): 측정할 작업
            
        Returns:
            Dict[str, Any]: 지연시간 측정 결과
        """
        try:
            if not self.execution_monitor:
                raise RuntimeError("실행 모니터가 초기화되지 않았습니다.")
                
            start_time = datetime.now()
            
            # 작업 실행
            if operation == 'order':
                result = await self.execute_order({
                    'symbol': 'BTC/USDT',
                    'side': 'BUY',
                    'quantity': 0.001
                })
            elif operation == 'cancel':
                result = await self.cancel_order('test_order_id')
            elif operation == 'position':
                result = await self.get_position('BTC/USDT')
            elif operation == 'market_data':
                result = await self.get_market_data('BTC/USDT')
            else:
                raise ValueError(f"지원하지 않는 작업: {operation}")
                
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            return {
                'success': True,
                'operation': operation,
                'latency': latency,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"지연시간 측정 실패: {str(e)}")
            raise

    async def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계 조회"""
        try:
            if not self.error_handler:
                raise RuntimeError("에러 핸들러가 초기화되지 않았습니다.")
                
            return await self.error_handler.get_error_stats()
            
        except Exception as e:
            self.logger.error(f"에러 통계 조회 실패: {str(e)}")
            raise

    async def _validate_order(self, order: Dict[str, Any]):
        """
        주문 검증
        
        Args:
            order (Dict[str, Any]): 주문 정보
        """
        try:
            # 필수 필드 검사
            required_fields = ['symbol', 'side', 'quantity']
            for field in required_fields:
                if field not in order:
                    raise ValueError(f"필수 필드 누락: {field}")
                    
            # 수량 검사
            if not isinstance(order['quantity'], (int, float)) or order['quantity'] <= 0:
                raise ValueError("잘못된 수량")
                
            # 거래 방향 검사
            if order['side'].upper() not in ['BUY', 'SELL']:
                raise ValueError("잘못된 거래 방향")
                
        except Exception as e:
            logger.error(f"주문 검증 실패: {str(e)}")
            raise

    async def _calculate_risk_exposure(self, order: Dict[str, Any]) -> float:
        """
        위험 노출 계산
        
        Args:
            order (Dict[str, Any]): 주문 정보
            
        Returns:
            float: 위험 노출
        """
        try:
            # 현재 포지션 조회
            position = await self.get_position(order['symbol'])
            
            # 포지션 크기
            current_size = position.get('size', 0.0)
            
            # 주문 방향에 따른 포지션 변화
            if order['side'].upper() == 'BUY':
                new_size = current_size + order['quantity']
            else:
                new_size = current_size - order['quantity']
                
            # 위험 노출 계산 (절대값)
            risk_exposure = abs(new_size)
            
            return risk_exposure
            
        except Exception as e:
            logger.error(f"위험 노출 계산 실패: {str(e)}")
            raise

    def _calculate_slippage(
        self,
        request: Dict[str, Any],
        result: Dict[str, Any]
    ) -> float:
        """
        슬리피지 계산
        
        Args:
            request (Dict[str, Any]): 주문 요청
            result (Dict[str, Any]): 실행 결과
            
        Returns:
            float: 슬리피지
        """
        try:
            # 요청 가격이 없는 경우 (시장가 주문)
            if 'price' not in request:
                return 0.0
                
            request_price = float(request['price'])
            execution_price = float(result.get('price', 0.0))
            
            if execution_price == 0.0:
                return 0.0
                
            # 슬리피지 계산
            side = request['side'].upper()
            if side == 'BUY':
                # 매수: 실행가 > 요청가인 경우 양수
                slippage = (execution_price - request_price) / request_price
            else:
                # 매도: 실행가 < 요청가인 경우 양수
                slippage = (request_price - execution_price) / request_price
                
            return slippage
            
        except Exception as e:
            logger.error(f"슬리피지 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_execution_cost(
        self,
        request: Dict[str, Any],
        result: Dict[str, Any]
    ) -> float:
        """
        실행 비용 계산
        
        Args:
            request (Dict[str, Any]): 주문 요청
            result (Dict[str, Any]): 실행 결과
            
        Returns:
            float: 실행 비용
        """
        try:
            # 기본 거래 수수료
            fee_rate = self.config.get('fee_rate', 0.001)
            
            # 실행 수량
            executed_quantity = float(result.get('executed_qty', 0.0))
            
            # 실행 가격
            execution_price = float(result.get('price', 0.0))
            
            # 비용 계산
            fee_cost = executed_quantity * execution_price * fee_rate
            
            # 슬리피지 비용
            slippage = self._calculate_slippage(request, result)
            slippage_cost = executed_quantity * execution_price * abs(slippage)
            
            # 총 비용
            total_cost = fee_cost + slippage_cost
            
            # 정규화된 비용 (거래 금액 대비)
            normalized_cost = total_cost / (executed_quantity * execution_price) if executed_quantity > 0 else 0.0
            
            return normalized_cost
            
        except Exception as e:
            logger.error(f"실행 비용 계산 실패: {str(e)}")
            return 0.0
            
    async def retry_execution(
        self,
        order_id: str,
        max_retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        실행 재시도
        
        Args:
            order_id (str): 주문 ID
            max_retries (Optional[int]): 최대 재시도 횟수
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # 기존 실행 확인
            if order_id not in self.active_executions:
                raise ValueError(f"존재하지 않는 주문 ID: {order_id}")
                
            execution_info = self.active_executions[order_id]
            request = execution_info['request']
            strategy = execution_info['strategy']
            
            # 재시도 횟수 설정
            retries = max_retries if max_retries is not None else self.max_retries
            retry_count = 0
            last_error = None
            
            # 재시도 로직
            while retry_count < retries:
                retry_count += 1
                
                try:
                    logger.info(f"주문 실행 재시도 ({retry_count}/{retries}): {order_id}")
                    
                    # 재시도 실행
                    result = await self.execute_order(request, strategy)
                    
                    if result.get('success', False):
                        return result
                        
                    last_error = result.get('error', 'Unknown error')
                    
                except Exception as e:
                    logger.error(f"재시도 중 오류 발생: {str(e)}")
                    last_error = str(e)
                    
                # 재시도 지연
                await asyncio.sleep(self.retry_delay)
                
            # 모든 재시도 실패
            logger.error(f"최대 재시도 횟수 초과: {order_id}")
            
            # 알림 전송
            await self.notifier.notify_error(
                error_type="retry_failure",
                message=f"최대 재시도 횟수 초과: {order_id}",
                details={
                    'order_id': order_id,
                    'retries': retry_count,
                    'last_error': last_error
                }
            )
            
            return {
                'success': False,
                'order_id': order_id,
                'error': f"최대 재시도 횟수 초과: {last_error}",
                'retries': retry_count
            }
            
        except Exception as e:
            logger.error(f"실행 재시도 실패: {str(e)}")
            return {
                'success': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    # 이하 실행 전략 구현
    
    async def _execute_twap(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """TWAP 전략으로 주문 실행"""
        # 테스트용 더미 구현
        return {
            'success': True,
            'order_id': order_request['order_id'],
            'symbol': order_request['symbol'],
            'order_type': order_request['order_type'],
            'side': order_request['side'],
            'quantity': order_request['quantity'],
            'price': order_request['price'],
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
    async def _execute_vwap(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """VWAP 전략으로 주문 실행"""
        # 테스트용 더미 구현
        return {
            'success': True,
            'order_id': order_request['order_id'],
            'symbol': order_request['symbol'],
            'order_type': order_request['order_type'],
            'side': order_request['side'],
            'quantity': order_request['quantity'],
            'price': order_request['price'],
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
    async def _execute_market(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """시장가 주문 실행"""
        # 테스트용 더미 구현
        return {
            'success': True,
            'order_id': order_request['order_id'],
            'symbol': order_request['symbol'],
            'order_type': 'market',
            'side': order_request['side'],
            'quantity': order_request['quantity'],
            'price': order_request.get('current_price', order_request['price']),
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
    async def _execute_limit(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """지정가 주문 실행"""
        # 테스트용 더미 구현
        return {
            'success': True,
            'order_id': order_request['order_id'],
            'symbol': order_request['symbol'],
            'order_type': 'limit',
            'side': order_request['side'],
            'quantity': order_request['quantity'],
            'price': order_request['price'],
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
    async def _execute_iceberg(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """아이스버그 주문 실행"""
        # 테스트용 더미 구현
        return {
            'success': True,
            'order_id': order_request['order_id'],
            'symbol': order_request['symbol'],
            'order_type': 'iceberg',
            'side': order_request['side'],
            'quantity': order_request['quantity'],
            'price': order_request['price'],
            'status': 'filled',
            'timestamp': datetime.now()
        }
        
    async def _execute_adaptive(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """적응형 주문 실행"""
        # 테스트용 더미 구현
        return {
            'success': True,
            'order_id': order_request['order_id'],
            'symbol': order_request['symbol'],
            'order_type': 'adaptive',
            'side': order_request['side'],
            'quantity': order_request['quantity'],
            'price': order_request['price'],
            'status': 'filled',
            'timestamp': datetime.now()
        }

    async def _generate_execution_id(self) -> str:
        """실행 ID 생성"""
        return f"execution_{int(datetime.now().timestamp())}"

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """주문 정보 조회"""
        try:
            if not self.exchange_client:
                raise RuntimeError("거래소 클라이언트가 초기화되지 않았습니다.")
            
            # 주문 정보 조회
            order = self.exchange_client.get_order(symbol, order_id)
            
            # 실행 정보 업데이트
            if order_id in self.active_executions:
                self.active_executions[order_id].update({
                    "status": order["status"],
                    "timestamp": order.get("transactTime", None)
                })
            
            return order
            
        except Exception as e:
            self.logger.error(f"주문 정보 조회 실패: {str(e)}")
            return None

    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """활성 실행 목록 조회"""
        return self.active_executions 