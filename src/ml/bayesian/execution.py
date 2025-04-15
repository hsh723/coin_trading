import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import json
import os
import asyncio
import websockets
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

class ExecutionStrategy(Enum):
    """실행 전략"""
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    SMART_ROUTING = "smart_routing"

class Order:
    """주문 클래스"""
    
    def __init__(self,
                 order_id: str,
                 symbol: str,
                 order_type: OrderType,
                 side: OrderSide,
                 quantity: float,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 limit_price: Optional[float] = None,
                 time_in_force: str = "GTC"):
        """
        주문 초기화
        
        Args:
            order_id: 주문 ID
            symbol: 심볼
            order_type: 주문 유형
            side: 주문 방향
            quantity: 수량
            price: 가격 (시장가 주문의 경우)
            stop_price: 스탑 가격
            limit_price: 지정가
            time_in_force: 주문 유효 기간
        """
        self.order_id = order_id
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.time_in_force = time_in_force
        
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def update_status(self,
                     status: OrderStatus,
                     filled_quantity: float,
                     fill_price: float) -> None:
        """
        주문 상태 업데이트
        
        Args:
            status: 새로운 상태
            filled_quantity: 체결 수량
            fill_price: 체결 가격
        """
        self.status = status
        self.filled_quantity = filled_quantity
        self.avg_fill_price = fill_price
        self.updated_at = datetime.now()
        
    def is_completed(self) -> bool:
        """주문 완료 여부 확인"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]
        
    def get_remaining_quantity(self) -> float:
        """남은 수량 계산"""
        return self.quantity - self.filled_quantity

class OrderManager:
    """주문 관리 시스템"""
    
    def __init__(self,
                 config_path: str = "./config/execution_config.json",
                 data_dir: str = "./data",
                 log_dir: str = "./logs"):
        """
        주문 관리 시스템 초기화
        
        Args:
            config_path: 설정 파일 경로
            data_dir: 데이터 디렉토리
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("order_manager")
        
        # 설정 로드
        self.config = self._load_config()
        
        # 주문 파라미터
        self.max_slippage = self.config.get("max_slippage", 0.001)
        self.min_order_size = self.config.get("min_order_size", 0.001)
        self.max_order_size = self.config.get("max_order_size", 100.0)
        
        # 주문 관리
        self.orders = {}
        self.order_history = []
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def create_order(self,
                    symbol: str,
                    order_type: OrderType,
                    side: OrderSide,
                    quantity: float,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    limit_price: Optional[float] = None,
                    time_in_force: str = "GTC") -> Optional[Order]:
        """
        주문 생성
        
        Args:
            symbol: 심볼
            order_type: 주문 유형
            side: 주문 방향
            quantity: 수량
            price: 가격
            stop_price: 스탑 가격
            limit_price: 지정가
            time_in_force: 주문 유효 기간
            
        Returns:
            생성된 주문 객체
        """
        try:
            # 수량 검증
            if quantity < self.min_order_size or quantity > self.max_order_size:
                self.logger.error(f"주문 수량이 허용 범위를 벗어났습니다: {quantity}")
                return None
                
            # 주문 ID 생성
            order_id = f"{symbol}_{order_type.value}_{side.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 주문 객체 생성
            order = Order(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                limit_price=limit_price,
                time_in_force=time_in_force
            )
            
            # 주문 저장
            self.orders[order_id] = order
            self.order_history.append(order)
            
            self.logger.info(f"주문 생성 완료: {order_id}")
            return order
            
        except Exception as e:
            self.logger.error(f"주문 생성 중 오류 발생: {e}")
            return None
            
    def execute_twap(self,
                    order: Order,
                    duration: int = 300,
                    num_slices: int = 10) -> List[Order]:
        """
        TWAP 전략으로 주문 실행
        
        Args:
            order: 주문 객체
            duration: 실행 기간 (초)
            num_slices: 분할 수
            
        Returns:
            실행된 주문 리스트
        """
        try:
            # 분할 주문 생성
            slice_quantity = order.quantity / num_slices
            slice_duration = duration / num_slices
            
            executed_orders = []
            for i in range(num_slices):
                # 분할 주문 생성
                slice_order = self.create_order(
                    symbol=order.symbol,
                    order_type=order.order_type,
                    side=order.side,
                    quantity=slice_quantity,
                    price=order.price,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    time_in_force=order.time_in_force
                )
                
                if slice_order:
                    executed_orders.append(slice_order)
                    
                # 다음 분할 주문까지 대기
                time.sleep(slice_duration)
                
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"TWAP 실행 중 오류 발생: {e}")
            return []
            
    def execute_vwap(self,
                    order: Order,
                    market_data: pd.DataFrame,
                    num_slices: int = 10) -> List[Order]:
        """
        VWAP 전략으로 주문 실행
        
        Args:
            order: 주문 객체
            market_data: 시장 데이터
            num_slices: 분할 수
            
        Returns:
            실행된 주문 리스트
        """
        try:
            # 거래량 가중치 계산
            volume_weights = market_data['volume'] / market_data['volume'].sum()
            
            # 분할 주문 생성
            executed_orders = []
            for i in range(num_slices):
                # 거래량 가중치에 따른 수량 계산
                slice_quantity = order.quantity * volume_weights.iloc[i]
                
                # 분할 주문 생성
                slice_order = self.create_order(
                    symbol=order.symbol,
                    order_type=order.order_type,
                    side=order.side,
                    quantity=slice_quantity,
                    price=order.price,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    time_in_force=order.time_in_force
                )
                
                if slice_order:
                    executed_orders.append(slice_order)
                    
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"VWAP 실행 중 오류 발생: {e}")
            return []
            
    def execute_iceberg(self,
                       order: Order,
                       visible_size: float,
                       num_slices: int = 10) -> List[Order]:
        """
        Iceberg 전략으로 주문 실행
        
        Args:
            order: 주문 객체
            visible_size: 표시 수량
            num_slices: 분할 수
            
        Returns:
            실행된 주문 리스트
        """
        try:
            # 분할 주문 생성
            slice_quantity = visible_size
            total_slices = int(np.ceil(order.quantity / visible_size))
            
            executed_orders = []
            for i in range(min(total_slices, num_slices)):
                # 마지막 분할 주문의 경우 남은 수량으로 조정
                if i == total_slices - 1:
                    slice_quantity = order.quantity - (i * visible_size)
                    
                # 분할 주문 생성
                slice_order = self.create_order(
                    symbol=order.symbol,
                    order_type=order.order_type,
                    side=order.side,
                    quantity=slice_quantity,
                    price=order.price,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    time_in_force=order.time_in_force
                )
                
                if slice_order:
                    executed_orders.append(slice_order)
                    
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"Iceberg 실행 중 오류 발생: {e}")
            return []
            
    def execute_smart_routing(self,
                            order: Order,
                            market_data: Dict[str, pd.DataFrame]) -> List[Order]:
        """
        Smart Routing 전략으로 주문 실행
        
        Args:
            order: 주문 객체
            market_data: 시장 데이터 딕셔너리
            
        Returns:
            실행된 주문 리스트
        """
        try:
            # 시장별 스프레드 계산
            spreads = {}
            for exchange, data in market_data.items():
                bid = data['bid'].iloc[-1]
                ask = data['ask'].iloc[-1]
                spreads[exchange] = (ask - bid) / bid
                
            # 최적의 거래소 선택
            best_exchange = min(spreads, key=spreads.get)
            
            # 주문 실행
            executed_order = self.create_order(
                symbol=order.symbol,
                order_type=order.order_type,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                limit_price=order.limit_price,
                time_in_force=order.time_in_force
            )
            
            return [executed_order] if executed_order else []
            
        except Exception as e:
            self.logger.error(f"Smart Routing 실행 중 오류 발생: {e}")
            return []
            
    def cancel_order(self, order_id: str) -> bool:
        """
        주문 취소
        
        Args:
            order_id: 주문 ID
            
        Returns:
            취소 성공 여부
        """
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                if not order.is_completed():
                    order.update_status(OrderStatus.CANCELED, order.filled_quantity, order.avg_fill_price)
                    self.logger.info(f"주문 취소 완료: {order_id}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"주문 취소 중 오류 발생: {e}")
            return False
            
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        주문 상태 조회
        
        Args:
            order_id: 주문 ID
            
        Returns:
            주문 상태
        """
        try:
            if order_id in self.orders:
                return self.orders[order_id].status
            return None
            
        except Exception as e:
            self.logger.error(f"주문 상태 조회 중 오류 발생: {e}")
            return None
            
    def get_order_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         symbol: Optional[str] = None) -> List[Order]:
        """
        주문 이력 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            symbol: 심볼
            
        Returns:
            주문 이력 리스트
        """
        try:
            filtered_orders = self.order_history
            
            # 시간 필터링
            if start_time:
                filtered_orders = [o for o in filtered_orders if o.created_at >= start_time]
            if end_time:
                filtered_orders = [o for o in filtered_orders if o.created_at <= end_time]
                
            # 심볼 필터링
            if symbol:
                filtered_orders = [o for o in filtered_orders if o.symbol == symbol]
                
            return filtered_orders
            
        except Exception as e:
            self.logger.error(f"주문 이력 조회 중 오류 발생: {e}")
            return []
            
    def calculate_slippage(self,
                         order: Order,
                         execution_price: float) -> float:
        """
        슬리피지 계산
        
        Args:
            order: 주문 객체
            execution_price: 실행 가격
            
        Returns:
            슬리피지
        """
        try:
            if order.price is None:
                return 0.0
                
            slippage = abs(execution_price - order.price) / order.price
            return slippage
            
        except Exception as e:
            self.logger.error(f"슬리피지 계산 중 오류 발생: {e}")
            return 0.0
            
    def generate_execution_report(self,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        실행 보고서 생성
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            실행 보고서 딕셔너리
        """
        try:
            # 주문 이력 필터링
            orders = self.get_order_history(start_time, end_time)
            
            # 실행 통계 계산
            total_orders = len(orders)
            filled_orders = len([o for o in orders if o.status == OrderStatus.FILLED])
            canceled_orders = len([o for o in orders if o.status == OrderStatus.CANCELED])
            rejected_orders = len([o for o in orders if o.status == OrderStatus.REJECTED])
            
            # 평균 슬리피지 계산
            slippages = [self.calculate_slippage(o, o.avg_fill_price) for o in orders if o.status == OrderStatus.FILLED]
            avg_slippage = np.mean(slippages) if slippages else 0.0
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_orders": total_orders,
                "filled_orders": filled_orders,
                "canceled_orders": canceled_orders,
                "rejected_orders": rejected_orders,
                "fill_rate": filled_orders / total_orders if total_orders > 0 else 0.0,
                "average_slippage": avg_slippage,
                "orders": [{
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "type": o.order_type.value,
                    "side": o.side.value,
                    "quantity": o.quantity,
                    "filled_quantity": o.filled_quantity,
                    "price": o.price,
                    "avg_fill_price": o.avg_fill_price,
                    "status": o.status.value,
                    "created_at": o.created_at.isoformat(),
                    "updated_at": o.updated_at.isoformat()
                } for o in orders]
            }
            
            # 보고서 저장
            self._save_execution_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"실행 보고서 생성 중 오류 발생: {e}")
            return {}
            
    def _save_execution_report(self, report: Dict[str, Any]) -> None:
        """
        실행 보고서 저장
        
        Args:
            report: 실행 보고서
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.data_dir, f"execution_report_{timestamp}.json")
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            self.logger.info(f"실행 보고서 저장 완료: {report_path}")
            
        except Exception as e:
            self.logger.error(f"실행 보고서 저장 중 오류 발생: {e}") 