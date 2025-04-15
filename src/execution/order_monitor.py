"""
실시간 주문 실행 모니터
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class OrderMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.update_interval = config.get('update_interval', 1)
        self.history_size = config.get('history_size', 1000)
        
        self.thresholds = {
            'latency': config.get('latency_threshold', 1000),  # ms
            'fill_rate': config.get('fill_rate_threshold', 0.95),
            'slippage': config.get('slippage_threshold', 0.001),
            'retry_limit': config.get('retry_limit', 3)
        }
        
        self.orders = {}
        self.order_history = []
        self.is_monitoring = False
        self.monitor_task = None
        
    async def initialize(self):
        try:
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitor_orders())
            logger.info("주문 모니터 초기화 완료")
        except Exception as e:
            logger.error(f"주문 모니터 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        try:
            self.is_monitoring = False
            if self.monitor_task:
                await self.monitor_task
            logger.info("주문 모니터 종료")
        except Exception as e:
            logger.error(f"주문 모니터 종료 실패: {str(e)}")
            
    async def _monitor_orders(self):
        try:
            while self.is_monitoring:
                await self._update_order_status()
                await asyncio.sleep(self.update_interval)
        except Exception as e:
            logger.error(f"주문 모니터링 실패: {str(e)}")
            
    async def _update_order_status(self):
        try:
            now = datetime.now()
            for order_id, order in list(self.orders.items()):
                await self._check_order_status(order_id, order, now)
        except Exception as e:
            logger.error(f"주문 상태 업데이트 실패: {str(e)}")
            
    async def _check_order_status(self, order_id: str, order: Dict, now: datetime):
        try:
            # 주문 상태 체크 및 업데이트
            elapsed_time = (now - order['timestamp']).total_seconds() * 1000
            
            if elapsed_time > self.thresholds['latency']:
                await self._handle_delayed_order(order_id, order)
                
            if order['retry_count'] >= self.thresholds['retry_limit']:
                await self._handle_failed_order(order_id, order)
                
            fill_rate = order.get('filled_quantity', 0) / order['quantity']
            if fill_rate < self.thresholds['fill_rate']:
                await self._handle_unfilled_order(order_id, order)
                
        except Exception as e:
            logger.error(f"주문 상태 체크 실패: {str(e)}")
            
    async def _handle_delayed_order(self, order_id: str, order: Dict):
        try:
            logger.warning(f"주문 지연 감지: {order_id}")
            order['status'] = 'delayed'
            await self._retry_order(order_id, order)
        except Exception as e:
            logger.error(f"지연 주문 처리 실패: {str(e)}")
            
    async def _handle_failed_order(self, order_id: str, order: Dict):
        try:
            logger.error(f"주문 실패: {order_id}")
            order['status'] = 'failed'
            self._archive_order(order_id)
        except Exception as e:
            logger.error(f"실패 주문 처리 실패: {str(e)}")
            
    async def _handle_unfilled_order(self, order_id: str, order: Dict):
        try:
            logger.warning(f"미체결 주문 감지: {order_id}")
            order['status'] = 'unfilled'
            await self._adjust_order(order_id, order)
        except Exception as e:
            logger.error(f"미체결 주문 처리 실패: {str(e)}")
            
    async def _retry_order(self, order_id: str, order: Dict):
        try:
            if order['retry_count'] < self.thresholds['retry_limit']:
                order['retry_count'] += 1
                order['timestamp'] = datetime.now()
                logger.info(f"주문 재시도: {order_id} (시도 {order['retry_count']})")
            else:
                await self._handle_failed_order(order_id, order)
        except Exception as e:
            logger.error(f"주문 재시도 실패: {str(e)}")
            
    async def _adjust_order(self, order_id: str, order: Dict):
        try:
            # 주문 조정 로직
            remaining_quantity = order['quantity'] - order.get('filled_quantity', 0)
            if remaining_quantity > 0:
                order['quantity'] = remaining_quantity
                order['timestamp'] = datetime.now()
                logger.info(f"주문 조정: {order_id} (잔량: {remaining_quantity})")
        except Exception as e:
            logger.error(f"주문 조정 실패: {str(e)}")
            
    def _archive_order(self, order_id: str):
        try:
            if order_id in self.orders:
                order = self.orders.pop(order_id)
                self.order_history.append(order)
                
                if len(self.order_history) > self.history_size:
                    self.order_history = self.order_history[-self.history_size:]
        except Exception as e:
            logger.error(f"주문 아카이브 실패: {str(e)}")
            
    def add_order(self, order_id: str, order: Dict):
        try:
            order.update({
                'timestamp': datetime.now(),
                'status': 'new',
                'retry_count': 0
            })
            self.orders[order_id] = order
            logger.info(f"신규 주문 추가: {order_id}")
        except Exception as e:
            logger.error(f"주문 추가 실패: {str(e)}")
            
    def update_order(self, order_id: str, updates: Dict):
        try:
            if order_id in self.orders:
                self.orders[order_id].update(updates)
                logger.info(f"주문 업데이트: {order_id}")
        except Exception as e:
            logger.error(f"주문 업데이트 실패: {str(e)}")
            
    def get_order(self, order_id: str) -> Optional[Dict]:
        return self.orders.get(order_id)
        
    def get_active_orders(self) -> List[Dict]:
        return list(self.orders.values())
        
    def get_order_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        try:
            filtered_history = self.order_history
            
            if start_time:
                filtered_history = [
                    o for o in filtered_history
                    if o['timestamp'] >= start_time
                ]
                
            if end_time:
                filtered_history = [
                    o for o in filtered_history
                    if o['timestamp'] <= end_time
                ]
                
            return filtered_history
            
        except Exception as e:
            logger.error(f"주문 이력 조회 실패: {str(e)}")
            return []
            
    def get_execution_statistics(self) -> Dict[str, float]:
        try:
            total_orders = len(self.order_history)
            if total_orders == 0:
                return {
                    'success_rate': 0.0,
                    'fill_rate': 0.0,
                    'avg_latency': 0.0,
                    'avg_slippage': 0.0
                }
                
            success_count = len([o for o in self.order_history if o['status'] == 'filled'])
            total_latency = sum(
                (o.get('filled_time', o['timestamp']) - o['timestamp']).total_seconds() * 1000
                for o in self.order_history if o['status'] == 'filled'
            )
            total_slippage = sum(
                abs(o.get('executed_price', 0) - o.get('target_price', 0)) / o.get('target_price', 1)
                for o in self.order_history if o['status'] == 'filled'
            )
            
            return {
                'success_rate': success_count / total_orders,
                'fill_rate': sum(o.get('filled_quantity', 0) for o in self.order_history) / sum(o['quantity'] for o in self.order_history),
                'avg_latency': total_latency / success_count if success_count > 0 else 0.0,
                'avg_slippage': total_slippage / success_count if success_count > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"실행 통계 계산 실패: {str(e)}")
            return {
                'success_rate': 0.0,
                'fill_rate': 0.0,
                'avg_latency': 0.0,
                'avg_slippage': 0.0
            } 