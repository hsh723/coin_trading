"""
VWAP 실행 전략 모듈

거래량 가중 평균 가격(Volume-Weighted Average Price) 전략을 구현합니다.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from src.execution.strategies.base_strategy import BaseExecutionStrategy

logger = logging.getLogger(__name__)

class VwapExecutionStrategy(BaseExecutionStrategy):
    """VWAP 실행 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        VWAP 실행 전략 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        super().__init__(config)
        
        # 전략 특정 설정
        self.time_window = config.get('time_window', 3600)  # 실행 시간 창(초)
        self.interval_count = config.get('interval_count', 12)  # 시간 간격 수
        self.volume_profile = config.get('volume_profile', None)  # 거래량 프로필
        self.deviation_limit = config.get('deviation_limit', 0.02)  # 추정 VWAP 대비 최대 편차
        self.min_participation_rate = config.get('min_participation_rate', 0.05)  # 최소 참여율
        self.max_participation_rate = config.get('max_participation_rate', 0.3)  # 최대 참여율
        
        # 기본 거래량 프로필 (없는 경우)
        if not self.volume_profile:
            self.volume_profile = self._create_default_volume_profile()
            
        # 실행 상태
        self.is_active = False
        self.execution_task = None
        self.start_time = None
        self.end_time = None
        self.interval_seconds = 0
        self.remaining_quantity = 0.0
        self.executed_quantity = 0.0
        self.order_side = ''
        self.symbol = ''
        self.execution_results = []
        self.volume_history = []
        self.estimated_vwap = 0.0
        
    def _create_default_volume_profile(self) -> List[float]:
        """
        기본 거래량 프로필 생성
        
        Returns:
            List[float]: 거래량 프로필 (합이 1.0)
        """
        # U자 형태의 거래량 프로필 (주로 주식 시장에서 관찰됨)
        if self.interval_count < 5:
            return [1.0 / self.interval_count] * self.interval_count
            
        profile = []
        for i in range(self.interval_count):
            # 시작과 끝에 더 많은 거래량
            position = i / (self.interval_count - 1)  # 0.0 ~ 1.0
            weight = 1.0 - 0.5 * np.sin(position * np.pi)
            profile.append(weight)
            
        # 정규화
        total = sum(profile)
        return [w / total for w in profile]
        
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
            self.volume_history = []
            
            # 시간 창 설정 (요청에 제공된 경우 사용)
            if 'time_window' in order_request:
                self.time_window = float(order_request['time_window'])
                
            # 간격 수 설정 (요청에 제공된 경우 사용)
            if 'interval_count' in order_request:
                self.interval_count = int(order_request['interval_count'])
                
            # 거래량 프로필 설정 (요청에 제공된 경우 사용)
            if 'volume_profile' in order_request:
                profile = order_request['volume_profile']
                # 정규화
                total = sum(profile)
                self.volume_profile = [p / total for p in profile]
            elif self.interval_count != len(self.volume_profile):
                # 간격 수와 프로필 길이가 다르면 기본 프로필 다시 생성
                self.volume_profile = self._create_default_volume_profile()
                
            # 시간 창이 너무 짧은 경우 조정
            if self.time_window < 60:
                self.time_window = 60  # 최소 1분
                
            # 시작/종료 시간 설정
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(seconds=self.time_window)
            
            # 간격 초 계산
            self.interval_seconds = self.time_window / self.interval_count
            
            # 활성화 상태로 설정
            self.is_active = True
            
            # 예상 VWAP 초기화
            self.estimated_vwap = self._get_current_price(order_request)
            
            # 분할 실행 시작
            logger.info(
                f"VWAP 전략 실행 시작: {self.symbol}, 수량: {self.remaining_quantity}, "
                f"시간 창: {self.time_window}초, 간격 수: {self.interval_count}"
            )
            
            self.execution_task = asyncio.create_task(self._execute_intervals())
            execution_result = await self.execution_task
            
            return execution_result
            
        except Exception as e:
            logger.error(f"VWAP 전략 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'vwap',
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
                
            logger.info(f"VWAP 전략 실행 취소: {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"VWAP 전략 취소 오류: {str(e)}")
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
            
    def _get_current_price(self, order_request: Dict[str, Any]) -> float:
        """
        현재 가격 조회
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
            
        Returns:
            float: 현재 가격
        """
        if 'current_price' in order_request and order_request['current_price'] > 0:
            return float(order_request['current_price'])
            
        # 오더북에서 조회
        if 'orderbook' in order_request and order_request['orderbook']:
            orderbook = order_request['orderbook']
            if 'asks' in orderbook and 'bids' in orderbook:
                if orderbook['asks'] and orderbook['bids']:
                    best_ask = float(orderbook['asks'][0][0])
                    best_bid = float(orderbook['bids'][0][0])
                    return (best_ask + best_bid) / 2
                    
        # 기본값
        return 100.0  # 샘플 가격
        
    async def _collect_volume_data(self) -> float:
        """
        거래량 데이터 수집 및 현재 볼륨 조회
        
        Returns:
            float: 현재 거래량
        """
        try:
            # TODO: 실제 구현에서는 실제 거래소 API 호출
            
            # 샘플 거래량 (실제로는 API에서 가져와야 함)
            now = datetime.now()
            elapsed_seconds = (now - self.start_time).total_seconds()
            interval_position = int(elapsed_seconds / self.interval_seconds)
            
            if 0 <= interval_position < len(self.volume_profile):
                # 거래량 프로필에 따른 예상 거래량
                relative_volume = self.volume_profile[interval_position]
            else:
                relative_volume = 0.1
                
            # 랜덤 요소 추가
            volume_noise = np.random.uniform(0.8, 1.2)
            current_volume = relative_volume * volume_noise * 100.0  # 샘플 거래량
            
            # 거래량 이력 기록
            self.volume_history.append((now, current_volume))
            
            return current_volume
            
        except Exception as e:
            logger.error(f"거래량 데이터 수집 오류: {str(e)}")
            return 10.0  # 기본 거래량
            
    async def _update_estimated_vwap(self) -> None:
        """예상 VWAP 업데이트"""
        try:
            # TODO: 실제 구현에서는 거래소 API에서 실제 VWAP 조회
            # 여기서는 간단히 최근 거래 가격의 평균으로 추정
            
            # 현재 가격 조회 (실제로는 API에서)
            current_price = 100.0  # 샘플 가격
            
            # 간단한 지수 이동 평균으로 VWAP 추정
            alpha = 0.2  # 가중치 (0.0 ~ 1.0)
            self.estimated_vwap = alpha * current_price + (1 - alpha) * self.estimated_vwap
            
        except Exception as e:
            logger.error(f"VWAP 업데이트 오류: {str(e)}")
            
    async def _execute_intervals(self) -> Dict[str, Any]:
        """
        간격별 주문 실행
        
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # 거래량에 따른 실행 비율 계산
            total_weight = sum(self.volume_profile)
            executed_weight = 0.0
            
            for i in range(self.interval_count):
                if not self.is_active or self.remaining_quantity <= 0:
                    break
                    
                # 다음 간격 시작 시간 계산
                next_interval_time = self.start_time + timedelta(seconds=i * self.interval_seconds)
                
                # 현재 시간이 다음 간격 시작 시간보다 앞서면 대기
                wait_time = (next_interval_time - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    
                # 현재 거래량 정보 수집
                current_volume = await self._collect_volume_data()
                
                # 예상 VWAP 업데이트
                await self._update_estimated_vwap()
                
                # 현재 간격의 거래량 가중치
                interval_weight = self.volume_profile[i]
                
                # 현재 간격까지의 누적 가중치 비율
                expected_cumulative_ratio = (executed_weight + interval_weight) / total_weight
                
                # 목표 누적 실행 수량
                target_executed_qty = self.remaining_quantity + self.executed_quantity
                target_cum_executed = target_executed_qty * expected_cumulative_ratio
                
                # 이번 간격에 실행해야 할 수량
                interval_target_qty = max(0.0, target_cum_executed - self.executed_quantity)
                
                # 거래량에 따른 참여율 조정
                participation_rate = self._calculate_participation_rate(current_volume)
                
                # 간격 내 분할 실행
                await self._execute_interval_slices(i, interval_target_qty, participation_rate)
                
                # 실행 후 누적 가중치 업데이트
                executed_weight += interval_weight
                
            # 남은 수량이 있으면 마무리 실행
            if self.is_active and self.remaining_quantity > 0:
                logger.info(f"VWAP 마무리 실행: 남은 수량 {self.remaining_quantity:.6f}")
                
                # 마지막 실행은 시장가로
                final_result = await self._execute_market_order(self.remaining_quantity)
                self.execution_results.append(final_result)
                
                if final_result['success']:
                    executed_qty = float(final_result.get('executed_qty', 0.0))
                    self.executed_quantity += executed_qty
                    self.remaining_quantity -= executed_qty
                    
            # 실행 완료
            execution_complete = (self.remaining_quantity <= 0)
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            # 실행 결과 집계
            result = {
                'success': execution_complete,
                'strategy': 'vwap',
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
                f"VWAP 전략 실행 완료: 성공={execution_complete}, "
                f"실행량={self.executed_quantity:.6f}, 평균가={result['average_price']:.2f}"
            )
            
            return result
            
        except asyncio.CancelledError:
            logger.info("VWAP 전략 실행 취소됨")
            return {
                'success': False,
                'error': 'cancelled',
                'strategy': 'vwap',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"VWAP 전략 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'vwap',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'timestamp': datetime.now()
            }
        finally:
            self.is_active = False
            
    def _calculate_participation_rate(self, current_volume: float) -> float:
        """
        거래량 참여율 계산
        
        Args:
            current_volume (float): 현재 거래량
            
        Returns:
            float: 참여율 (0.0 ~ 1.0)
        """
        # 기본 참여율
        base_rate = (self.min_participation_rate + self.max_participation_rate) / 2
        
        # 볼륨이 충분히 크면 참여율 높임
        if current_volume > 0:
            order_size = self.remaining_quantity + self.executed_quantity
            volume_ratio = order_size / current_volume
            
            if volume_ratio < 0.05:
                # 주문이 거래량 대비 작으면 참여율 높임
                adjusted_rate = base_rate * 1.5
            elif volume_ratio > 0.2:
                # 주문이 거래량 대비 크면 참여율 낮춤
                adjusted_rate = base_rate * 0.7
            else:
                adjusted_rate = base_rate
        else:
            adjusted_rate = base_rate
            
        # 제한 범위 내에서 설정
        return max(self.min_participation_rate, min(adjusted_rate, self.max_participation_rate))
        
    async def _execute_interval_slices(
        self,
        interval_idx: int,
        target_qty: float,
        participation_rate: float
    ) -> None:
        """
        간격 내 분할 실행
        
        Args:
            interval_idx (int): 간격 인덱스
            target_qty (float): 목표 실행 수량
            participation_rate (float): 참여율
        """
        try:
            # 간격 내 분할 횟수 계산
            slice_count = max(1, min(5, int(target_qty / 0.01)))  # 최소 1, 최대 5회
            
            # 조각별 크기 계산
            slice_sizes = self._calculate_slice_sizes(target_qty, slice_count)
            
            # 간격 내 시간 분할
            slice_interval = self.interval_seconds / slice_count
            
            for i, size in enumerate(slice_sizes):
                if not self.is_active or size <= 0 or self.remaining_quantity <= 0:
                    break
                    
                # 실제 실행 수량 (남은 수량 제한)
                execution_qty = min(size, self.remaining_quantity)
                
                # 현재 시장 상태에 따라 실행 방법 선택
                await self._update_estimated_vwap()
                execution_method = self._select_execution_method()
                
                # 주문 실행
                if execution_method == 'market':
                    result = await self._execute_market_order(execution_qty)
                elif execution_method == 'limit':
                    limit_price = self._calculate_limit_price()
                    result = await self._execute_limit_order(execution_qty, limit_price)
                else:
                    # 기본 방법
                    result = await self._execute_market_order(execution_qty)
                    
                self.execution_results.append(result)
                
                if result['success']:
                    executed_qty = float(result.get('executed_qty', 0.0))
                    self.executed_quantity += executed_qty
                    self.remaining_quantity -= executed_qty
                    
                    logger.debug(
                        f"VWAP 간격 {interval_idx+1}/{self.interval_count} 조각 {i+1}/{slice_count} "
                        f"실행 완료: {executed_qty:.6f}, 남은 수량: {self.remaining_quantity:.6f}"
                    )
                else:
                    logger.warning(f"VWAP 간격 {interval_idx+1} 조각 {i+1} 실행 실패: {result.get('error', 'unknown error')}")
                    
                # 다음 조각까지 대기
                if i < len(slice_sizes) - 1:
                    await asyncio.sleep(slice_interval)
                    
        except Exception as e:
            logger.error(f"간격 내 분할 실행 오류: {str(e)}")
            
    def _calculate_slice_sizes(self, total_qty: float, slice_count: int) -> List[float]:
        """
        조각 크기 계산
        
        Args:
            total_qty (float): 총 수량
            slice_count (int): 조각 수
            
        Returns:
            List[float]: 조각별 크기
        """
        if slice_count <= 1:
            return [total_qty]
            
        # 약간의 랜덤성 추가
        noise = np.random.uniform(0.8, 1.2, slice_count)
        noise = noise / noise.sum()  # 정규화
        
        # 각 조각 크기 계산
        sizes = noise * total_qty
        
        # 합이 정확히 total_qty가 되도록 조정
        sizes[-1] = total_qty - sum(sizes[:-1])
        
        return sizes
        
    def _select_execution_method(self) -> str:
        """
        실행 방법 선택
        
        Returns:
            str: 실행 방법 ('market' 또는 'limit')
        """
        # 추정 VWAP에 기반한 실행 방법 선택
        current_price = 100.0  # 실제로는 API에서 현재가 조회
        
        if self.estimated_vwap <= 0:
            return 'market'
            
        price_deviation = abs((current_price - self.estimated_vwap) / self.estimated_vwap)
        
        # 현재 가격이 추정 VWAP에 가까우면 시장가 주문
        if price_deviation < self.deviation_limit:
            return 'market'
            
        # 그렇지 않으면 지정가 주문
        return 'limit'
        
    def _calculate_limit_price(self) -> float:
        """
        지정가 계산
        
        Returns:
            float: 지정가
        """
        # 추정 VWAP에 기반한 지정가 계산
        side = self.order_side.upper()
        
        if side == 'BUY':
            # 매수 시 추정 VWAP보다 약간 낮게
            return self.estimated_vwap * 0.9995
        else:
            # 매도 시 추정 VWAP보다 약간 높게
            return self.estimated_vwap * 1.0005
            
    async def _execute_market_order(self, quantity: float) -> Dict[str, Any]:
        """
        시장가 주문 실행
        
        Args:
            quantity (float): 실행 수량
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # TODO: 실제 구현에서는 거래소 API 호출
            
            # 성공적인 실행으로 가정한 샘플 응답
            current_price = 100.0  # 실제로는 API에서 현재가 조회
            
            return {
                'success': True,
                'strategy': 'market',
                'executed_qty': quantity,
                'price': current_price,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"시장가 주문 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def _execute_limit_order(self, quantity: float, price: float) -> Dict[str, Any]:
        """
        지정가 주문 실행
        
        Args:
            quantity (float): 실행 수량
            price (float): 지정가
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        try:
            # TODO: 실제 구현에서는 거래소 API 호출
            
            # 성공적인 실행으로 가정한 샘플 응답
            return {
                'success': True,
                'strategy': 'limit',
                'executed_qty': quantity,
                'price': price,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"지정가 주문 실행 오류: {str(e)}")
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