"""
적응형 실행 전략 모듈

시장 상황에 따라 실시간으로 전략을 조정하는 적응형 실행 전략입니다.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from src.execution.strategies.base_strategy import BaseExecutionStrategy

logger = logging.getLogger(__name__)

class AdaptiveExecutionStrategy(BaseExecutionStrategy):
    """적응형 실행 전략"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        적응형 실행 전략 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        super().__init__(config)
        
        # 전략 특정 설정
        self.slippage_threshold = config.get('slippage_threshold', 0.002)  # 0.2%
        self.volatility_threshold = config.get('volatility_threshold', 0.01)  # 1%
        self.max_participation_rate = config.get('max_participation_rate', 0.3)  # 최대 30%
        self.initial_participation_rate = config.get('initial_participation_rate', 0.1)  # 초기 10%
        self.urgency_factor = config.get('urgency_factor', 0.5)  # 0.0(느림) ~ 1.0(빠름)
        
        # 시장 상태 추적
        self.last_prices = []
        self.current_participation_rate = self.initial_participation_rate
        self.current_strategy = 'passive'  # passive, neutral, aggressive
        
        # 실행 상태
        self.is_active = False
        self.execution_task = None
        self.next_evaluation_time = datetime.now()
        self.remaining_quantity = 0.0
        self.executed_quantity = 0.0
        self.start_price = 0.0
        self.last_execution_price = 0.0
        self.market_volume = 0.0
        self.order_side = ''
        
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
            
            # 초기 설정
            self.symbol = order_request['symbol']
            self.order_side = order_request['side']
            self.remaining_quantity = float(order_request['quantity'])
            self.executed_quantity = 0.0
            self.start_price = order_request.get('current_price', 0.0)
            self.last_execution_price = self.start_price
            self.market_volume = self._estimate_market_volume(order_request)
            self.is_active = True
            
            # 긴급도 설정
            self.urgency_factor = order_request.get('urgency_factor', self.urgency_factor)
            
            # 현재 시장 상태 평가
            await self._evaluate_market_conditions(order_request)
            
            # 실행 전략 및 참여율 초기화
            self._adjust_execution_parameters()
            
            # 실행 시작
            logger.info(f"적응형 전략 실행 시작: {self.symbol}, 수량: {self.remaining_quantity}")
            
            # 분할 실행 (비동기)
            self.execution_task = asyncio.create_task(self._execute_order_slices())
            execution_result = await self.execution_task
            
            return execution_result
            
        except Exception as e:
            logger.error(f"적응형 전략 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'adaptive',
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
                
            logger.info(f"적응형 전략 실행 취소: {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"적응형 전략 취소 오류: {str(e)}")
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
            
    def _estimate_market_volume(self, order_request: Dict[str, Any]) -> float:
        """
        시장 거래량 추정
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
            
        Returns:
            float: 추정 시장 거래량
        """
        # 요청에 시장 거래량이 있는 경우 사용
        if 'market_volume' in order_request:
            return float(order_request['market_volume'])
            
        # 오더북 기반 거래량 추정
        if 'orderbook' in order_request:
            orderbook = order_request['orderbook']
            if orderbook and 'asks' in orderbook and 'bids' in orderbook:
                asks_volume = sum(float(ask[1]) for ask in orderbook['asks'][:5])
                bids_volume = sum(float(bid[1]) for bid in orderbook['bids'][:5])
                return max(asks_volume, bids_volume)
                
        # 기본값
        return float(order_request['quantity']) * 20  # 주문 수량의 20배로 가정
        
    async def _evaluate_market_conditions(self, order_request: Dict[str, Any]) -> None:
        """
        시장 상태 평가
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
        """
        # 현재가 기록
        current_price = order_request.get('current_price', 0.0)
        if current_price > 0:
            self.last_prices.append(current_price)
            
        # 최대 100개 가격 이력 유지
        if len(self.last_prices) > 100:
            self.last_prices = self.last_prices[-100:]
            
        # 변동성 계산 (충분한 데이터가 있는 경우)
        if len(self.last_prices) >= 3:
            self.current_volatility = self._calculate_volatility()
        else:
            self.current_volatility = 0.0
            
        # 스프레드 계산
        self.current_spread = self._calculate_spread(order_request)
        
        # 시장 유동성 평가
        self.current_liquidity = self._evaluate_liquidity(order_request)
        
        # 시장 추세 평가
        self.current_trend = self._calculate_trend()
        
        # 다음 시장 평가 시간 설정
        self.next_evaluation_time = datetime.now() + timedelta(seconds=10)
        
    def _calculate_volatility(self) -> float:
        """
        가격 변동성 계산
        
        Returns:
            float: 변동성 (표준 편차 / 평균)
        """
        if len(self.last_prices) < 2:
            return 0.0
            
        prices = np.array(self.last_prices)
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        volatility = np.std(returns)
        
        return volatility
        
    def _calculate_spread(self, order_request: Dict[str, Any]) -> float:
        """
        스프레드 계산
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
            
        Returns:
            float: 스프레드 (%)
        """
        # 오더북이 있는 경우
        if 'orderbook' in order_request:
            orderbook = order_request['orderbook']
            if orderbook and 'asks' in orderbook and 'bids' in orderbook:
                if orderbook['asks'] and orderbook['bids']:
                    best_ask = float(orderbook['asks'][0][0])
                    best_bid = float(orderbook['bids'][0][0])
                    return (best_ask - best_bid) / best_bid
                    
        # 기본값
        return 0.001  # 0.1%
        
    def _evaluate_liquidity(self, order_request: Dict[str, Any]) -> float:
        """
        시장 유동성 평가
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
            
        Returns:
            float: 유동성 점수 (0.0 ~ 1.0)
        """
        # 오더북 기반 유동성 평가
        if 'orderbook' in order_request:
            orderbook = order_request['orderbook']
            if orderbook and 'asks' in orderbook and 'bids' in orderbook:
                # 주문 수량에 대한 주문 깊이(depth) 평가
                total_quantity = float(order_request['quantity'])
                side = order_request['side'].upper()
                
                if side == 'BUY':
                    # 매수 시 매도 주문(asks) 평가
                    available_volume = sum(float(ask[1]) for ask in orderbook['asks'][:10])
                else:
                    # 매도 시 매수 주문(bids) 평가
                    available_volume = sum(float(bid[1]) for bid in orderbook['bids'][:10])
                    
                # 유동성 점수 계산
                if total_quantity > 0:
                    liquidity_score = min(1.0, available_volume / total_quantity)
                    return liquidity_score
                    
        # 기본값
        return 0.5  # 중간 정도의 유동성
        
    def _calculate_trend(self) -> float:
        """
        시장 추세 계산
        
        Returns:
            float: 추세 점수 (-1.0 ~ 1.0, 양수=상승, 음수=하락)
        """
        if len(self.last_prices) < 3:
            return 0.0
            
        # 간단한 추세 계산
        prices = np.array(self.last_prices)
        price_diff = (prices[-1] - prices[0]) / prices[0]
        
        # 추세 강도 정규화 (-1 ~ 1)
        trend_score = max(min(price_diff * 50, 1.0), -1.0)
        
        return trend_score
        
    def _adjust_execution_parameters(self) -> None:
        """실행 파라미터 조정"""
        # 시장 상태에 따른 전략 조정
        
        # 변동성 기반 조정
        if self.current_volatility > self.volatility_threshold:
            # 변동성이 높은 경우 보수적 접근
            self.current_strategy = 'passive'
            volatility_factor = 0.5  # 참여율 감소
        else:
            # 변동성이 낮은 경우 중립적 접근
            self.current_strategy = 'neutral'
            volatility_factor = 1.0  # 참여율 유지
            
        # 스프레드 기반 조정
        if self.current_spread > self.slippage_threshold:
            # 스프레드가 넓은 경우 보수적 접근
            spread_factor = 0.7  # 참여율 감소
        else:
            # 스프레드가 좁은 경우 적극적 접근
            spread_factor = 1.2  # 참여율 증가
            
        # 유동성 기반 조정
        liquidity_factor = 0.5 + self.current_liquidity
        
        # 추세 기반 조정
        side = self.order_side.upper()
        trend_factor = 1.0
        
        if side == 'BUY' and self.current_trend > 0.3:
            # 매수 시 강한 상승 추세면 적극적 접근
            trend_factor = 1.3
            self.current_strategy = 'aggressive'
        elif side == 'SELL' and self.current_trend < -0.3:
            # 매도 시 강한 하락 추세면 적극적 접근
            trend_factor = 1.3
            self.current_strategy = 'aggressive'
        elif (side == 'BUY' and self.current_trend < -0.3) or (side == 'SELL' and self.current_trend > 0.3):
            # 추세 반대 시 보수적 접근
            trend_factor = 0.7
            self.current_strategy = 'passive'
            
        # 긴급도 요소 고려
        urgency_factor = 0.5 + self.urgency_factor * 0.5
        
        # 최종 참여율 계산
        base_rate = self.initial_participation_rate
        adjusted_rate = base_rate * volatility_factor * spread_factor * liquidity_factor * trend_factor * urgency_factor
        
        # 제한 범위 내에서 설정
        self.current_participation_rate = max(0.01, min(adjusted_rate, self.max_participation_rate))
        
        logger.debug(
            f"전략 조정: {self.current_strategy}, 참여율: {self.current_participation_rate:.2%}, "
            f"변동성: {self.current_volatility:.4f}, 스프레드: {self.current_spread:.4f}, "
            f"유동성: {self.current_liquidity:.2f}, 추세: {self.current_trend:.2f}"
        )
        
    async def _execute_order_slices(self) -> Dict[str, Any]:
        """
        분할 주문 실행
        
        Returns:
            Dict[str, Any]: 실행 결과
        """
        start_time = datetime.now()
        execution_complete = False
        execution_results = []
        
        try:
            while self.is_active and self.remaining_quantity > 0:
                # 시장 상태 재평가 (주기적)
                if datetime.now() >= self.next_evaluation_time:
                    await self._re_evaluate_market()
                    
                # 현재 참여율에 따른 실행 수량 계산
                slice_quantity = self._calculate_slice_quantity()
                
                if slice_quantity <= 0:
                    await asyncio.sleep(1.0)
                    continue
                    
                # 실행 전략 선택
                execution_strategy = self._select_execution_method()
                
                # 주문 실행
                slice_result = await self._execute_slice(slice_quantity, execution_strategy)
                execution_results.append(slice_result)
                
                if slice_result['success']:
                    executed_qty = float(slice_result.get('executed_qty', 0.0))
                    self.executed_quantity += executed_qty
                    self.remaining_quantity -= executed_qty
                    
                    if 'price' in slice_result:
                        self.last_execution_price = float(slice_result['price'])
                        
                    logger.debug(
                        f"조각 실행 완료: {executed_qty:.6f}, 남은 수량: {self.remaining_quantity:.6f}, "
                        f"방법: {execution_strategy}, 가격: {self.last_execution_price}"
                    )
                else:
                    logger.warning(f"조각 실행 실패: {slice_result.get('error', 'unknown error')}")
                    
                # 지연 (현재 전략에 따라 조정)
                delay = self._calculate_delay()
                await asyncio.sleep(delay)
                
            # 실행 완료
            execution_complete = (self.remaining_quantity <= 0)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 실행 결과 집계
            result = {
                'success': execution_complete,
                'strategy': 'adaptive',
                'sub_strategies': [result.get('strategy', 'unknown') for result in execution_results],
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'average_price': self._calculate_average_price(execution_results),
                'execution_time': execution_time,
                'slippage': self._calculate_slippage(),
                'slice_count': len(execution_results),
                'timestamp': datetime.now()
            }
            
            if not execution_complete:
                result['error'] = "실행 미완료"
                
            logger.info(
                f"적응형 전략 실행 완료: 성공={execution_complete}, "
                f"실행량={self.executed_quantity:.6f}, 평균가={result['average_price']:.2f}"
            )
            
            return result
            
        except asyncio.CancelledError:
            logger.info("적응형 전략 실행 취소됨")
            return {
                'success': False,
                'error': 'cancelled',
                'strategy': 'adaptive',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"적응형 전략 실행 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'strategy': 'adaptive',
                'executed_quantity': self.executed_quantity,
                'remaining_quantity': self.remaining_quantity,
                'timestamp': datetime.now()
            }
        finally:
            self.is_active = False
            
    async def _re_evaluate_market(self) -> None:
        """시장 상태 재평가"""
        try:
            # 최신 시장 데이터 조회
            market_data = await self._fetch_market_data()
            
            if market_data:
                # 가격 업데이트
                if 'price' in market_data and market_data['price'] > 0:
                    self.last_prices.append(market_data['price'])
                    
                    # 최대 100개 유지
                    if len(self.last_prices) > 100:
                        self.last_prices = self.last_prices[-100:]
                        
                # 시장 상태 평가
                self._evaluate_market_conditions_from_data(market_data)
                
                # 실행 전략 및 파라미터 조정
                self._adjust_execution_parameters()
                
            # 다음 평가 시간 설정
            self.next_evaluation_time = datetime.now() + timedelta(seconds=10)
            
        except Exception as e:
            logger.error(f"시장 재평가 오류: {str(e)}")
            
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """
        최신 시장 데이터 조회
        
        Returns:
            Dict[str, Any]: 시장 데이터
        """
        # TODO: 실제 구현에서는 거래소 API 또는 캐시에서 최신 데이터 조회
        return {
            'price': self.last_execution_price,
            'orderbook': None,
            'volume': self.market_volume
        }
        
    def _evaluate_market_conditions_from_data(self, market_data: Dict[str, Any]) -> None:
        """
        시장 데이터 기반 상태 평가
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
        """
        # 변동성 계산
        if len(self.last_prices) >= 3:
            self.current_volatility = self._calculate_volatility()
        else:
            self.current_volatility = 0.0
            
        # 스프레드 계산
        if 'orderbook' in market_data and market_data['orderbook']:
            orderbook = market_data['orderbook']
            if 'asks' in orderbook and 'bids' in orderbook:
                if orderbook['asks'] and orderbook['bids']:
                    best_ask = float(orderbook['asks'][0][0])
                    best_bid = float(orderbook['bids'][0][0])
                    self.current_spread = (best_ask - best_bid) / best_bid
                    
        # 시장 추세 평가
        self.current_trend = self._calculate_trend()
        
    def _calculate_slice_quantity(self) -> float:
        """
        실행 조각 수량 계산
        
        Returns:
            float: 조각 수량
        """
        # 참여율에 따른 기본 조각 크기
        slice_quantity = self.market_volume * self.current_participation_rate
        
        # 남은 수량 기준 제한
        slice_quantity = min(slice_quantity, self.remaining_quantity)
        
        # 현재 전략에 따른 조정
        if self.current_strategy == 'aggressive':
            slice_quantity *= 1.5
        elif self.current_strategy == 'passive':
            slice_quantity *= 0.5
            
        # 최소/최대 제한
        min_slice = min(0.001, self.remaining_quantity)  # 최소 0.001 또는 남은 수량
        max_slice = min(self.remaining_quantity, self.market_volume * self.max_participation_rate)
        
        return max(min_slice, min(slice_quantity, max_slice))
        
    def _select_execution_method(self) -> str:
        """
        실행 방법 선택
        
        Returns:
            str: 실행 방법 (market, limit, iceberg 등)
        """
        # 현재 전략에 따른 메서드 선택
        if self.current_strategy == 'aggressive':
            # 변동성, 스프레드에 따른 추가 결정
            if self.current_volatility > self.volatility_threshold * 2:
                return 'iceberg'  # 변동성이 매우 높은 경우 아이스버그
            return 'market'  # 빠른 실행 우선
        elif self.current_strategy == 'passive':
            # 낮은 변동성, 좁은 스프레드인 경우에만 지정가
            if self.current_spread < self.slippage_threshold:
                return 'limit'
            return 'iceberg'  # 그외 아이스버그
        else:  # neutral
            if self.current_spread > self.slippage_threshold:
                return 'iceberg'
            # 현재 시장 상황에 따라 다양하게 선택
            return np.random.choice(['market', 'limit'], p=[0.3, 0.7])
            
    def _calculate_delay(self) -> float:
        """
        실행 간 지연 계산
        
        Returns:
            float: 지연 시간(초)
        """
        # 전략별 기본 지연
        if self.current_strategy == 'aggressive':
            base_delay = 1.0
        elif self.current_strategy == 'passive':
            base_delay = 5.0
        else:  # neutral
            base_delay = 3.0
            
        # 변동성 기반 조정
        volatility_factor = 1.0 + max(0, self.current_volatility - self.volatility_threshold) * 10
        
        # 긴급도 반영
        urgency_factor = 2.0 - self.urgency_factor
        
        # 최종 지연 계산
        delay = base_delay * volatility_factor * urgency_factor
        
        # 제한 범위 내 설정
        return max(0.5, min(delay, 10.0))
        
    async def _execute_slice(
        self,
        quantity: float,
        method: str
    ) -> Dict[str, Any]:
        """
        주문 조각 실행
        
        Args:
            quantity (float): 실행 수량
            method (str): 실행 방법
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        # TODO: 실제 구현에서는 실제 거래 실행 로직
        current_price = self.last_execution_price
        
        # 성공적인 실행으로 가정한 샘플 응답
        return {
            'success': True,
            'strategy': method,
            'executed_qty': quantity,
            'price': current_price,
            'timestamp': datetime.now()
        }
        
    def _calculate_average_price(self, execution_results: List[Dict[str, Any]]) -> float:
        """
        평균 실행 가격 계산
        
        Args:
            execution_results (List[Dict[str, Any]]): 실행 결과 목록
            
        Returns:
            float: 평균 실행 가격
        """
        total_qty = 0.0
        total_value = 0.0
        
        for result in execution_results:
            if result.get('success', False):
                qty = float(result.get('executed_qty', 0.0))
                price = float(result.get('price', 0.0))
                
                if qty > 0 and price > 0:
                    total_qty += qty
                    total_value += qty * price
                    
        if total_qty > 0:
            return total_value / total_qty
        return 0.0
        
    def _calculate_slippage(self) -> float:
        """
        슬리피지 계산
        
        Returns:
            float: 슬리피지 (시작가 대비 평균 실행가의 변화율)
        """
        if self.start_price <= 0 or self.executed_quantity <= 0:
            return 0.0
            
        # 평균 실행가 계산 (임시, 실제로는 실행 결과를 기반으로 계산해야 함)
        avg_execution_price = self.last_execution_price
        
        # 슬리피지 계산
        side = self.order_side.upper()
        if side == 'BUY':
            # 매수: 실행가 > 요청가인 경우 양수
            slippage = (avg_execution_price - self.start_price) / self.start_price
        else:
            # 매도: 실행가 < 요청가인 경우 양수
            slippage = (self.start_price - avg_execution_price) / self.start_price
            
        return slippage 