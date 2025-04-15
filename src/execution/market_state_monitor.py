"""
실시간 시장 상태 모니터

이 모듈은 시장의 상태를 실시간으로 모니터링하여 실행 전략 최적화에 활용합니다.
주요 기능:
- 변동성 모니터링
- 유동성 분석
- 스프레드 추적
- 시장 상태 분류
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class MarketStateMonitor:
    def __init__(self, config: Dict[str, Any]):
        """
        시장 상태 모니터 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self.update_interval = int(config.get('update_interval', 1))
        self.window_size = int(config.get('window_size', 100))
        
        # 임계값 설정
        self.thresholds = {
            'high_volatility': float(config.get('volatility_threshold', 0.002)),
            'wide_spread': float(config.get('spread_threshold', 0.001)),
            'low_liquidity': float(config.get('liquidity_threshold', 10.0))
        }
        
        # 메트릭 저장소
        self.metrics = {
            'volatility': deque(maxlen=self.window_size),
            'spread': deque(maxlen=self.window_size),
            'liquidity': deque(maxlen=self.window_size),
            'volume': deque(maxlen=self.window_size),
            'timestamp': deque(maxlen=self.window_size)
        }
        
        # 시장 상태
        self.market_state = 'normal'
        self.is_monitoring = False
        self.monitor_task = None
        
    async def initialize(self):
        """모니터 초기화"""
        try:
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitor_market())
            logger.info("시장 상태 모니터 초기화 완료")
        except Exception as e:
            logger.error(f"시장 상태 모니터 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """모니터 종료"""
        try:
            self.is_monitoring = False
            if self.monitor_task:
                await self.monitor_task
            logger.info("시장 상태 모니터 종료")
        except Exception as e:
            logger.error(f"시장 상태 모니터 종료 실패: {str(e)}")
            
    async def _monitor_market(self):
        """시장 상태 모니터링"""
        try:
            while self.is_monitoring:
                await self._update_market_state()
                await asyncio.sleep(self.update_interval)
        except Exception as e:
            logger.error(f"시장 상태 모니터링 실패: {str(e)}")
            
    async def _update_market_state(self):
        """시장 상태 업데이트"""
        try:
            if not self.metrics['volatility']:
                return
                
            # 현재 메트릭 계산
            current_metrics = self._calculate_current_metrics()
            
            # 시장 상태 분류
            new_state = self._classify_market_state(current_metrics)
            
            if new_state != self.market_state:
                logger.info(f"시장 상태 변경: {self.market_state} -> {new_state}")
                self.market_state = new_state
                
        except Exception as e:
            logger.error(f"시장 상태 업데이트 실패: {str(e)}")
            
    def update_metrics(
        self,
        bid_price: float,
        ask_price: float,
        last_price: float,
        volume: float,
        timestamp: datetime
    ):
        """
        시장 메트릭 업데이트
        
        Args:
            bid_price (float): 매수 호가
            ask_price (float): 매도 호가
            last_price (float): 마지막 거래가
            volume (float): 거래량
            timestamp (datetime): 타임스탬프
        """
        try:
            # 스프레드 계산
            spread = (ask_price - bid_price) / bid_price
            
            # 변동성 계산
            volatility = self._calculate_volatility(last_price)
            
            # 유동성 계산
            liquidity = volume * last_price
            
            # 메트릭 저장
            self.metrics['spread'].append(spread)
            self.metrics['volatility'].append(volatility)
            self.metrics['liquidity'].append(liquidity)
            self.metrics['volume'].append(volume)
            self.metrics['timestamp'].append(timestamp)
            
            logger.debug(f"시장 메트릭 업데이트: spread={spread:.6f}, volatility={volatility:.6f}, liquidity={liquidity:.2f}")
            
        except Exception as e:
            logger.error(f"시장 메트릭 업데이트 실패: {str(e)}")
            
    def _calculate_volatility(self, current_price: float) -> float:
        """
        변동성 계산
        
        Args:
            current_price (float): 현재 가격
            
        Returns:
            float: 변동성
        """
        try:
            if not self.metrics['volatility']:
                return 0.0
                
            # 수익률 계산
            returns = []
            prices = list(self.metrics['volatility'])
            prices.append(current_price)
            
            for i in range(1, len(prices)):
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
                
            # 변동성 계산 (표준편차)
            return np.std(returns) if returns else 0.0
            
        except Exception as e:
            logger.error(f"변동성 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_current_metrics(self) -> Dict[str, float]:
        """
        현재 메트릭 계산
        
        Returns:
            Dict[str, float]: 현재 메트릭
        """
        try:
            if not self.metrics['volatility']:
                return {
                    'volatility': 0.0,
                    'spread': 0.0,
                    'liquidity': 0.0,
                    'volume': 0.0
                }
                
            return {
                'volatility': self.metrics['volatility'][-1],
                'spread': self.metrics['spread'][-1],
                'liquidity': self.metrics['liquidity'][-1],
                'volume': self.metrics['volume'][-1]
            }
            
        except Exception as e:
            logger.error(f"현재 메트릭 계산 실패: {str(e)}")
            return {
                'volatility': 0.0,
                'spread': 0.0,
                'liquidity': 0.0,
                'volume': 0.0
            }
            
    def _classify_market_state(self, metrics: Dict[str, float]) -> str:
        """
        시장 상태 분류
        
        Args:
            metrics (Dict[str, float]): 현재 메트릭
            
        Returns:
            str: 시장 상태
        """
        try:
            # 변동성 체크
            is_volatile = metrics['volatility'] > self.thresholds['high_volatility']
            
            # 스프레드 체크
            is_wide_spread = metrics['spread'] > self.thresholds['wide_spread']
            
            # 유동성 체크
            is_low_liquidity = metrics['liquidity'] < self.thresholds['low_liquidity']
            
            # 시장 상태 분류
            if is_volatile and is_wide_spread:
                return 'turbulent'
            elif is_volatile:
                return 'volatile'
            elif is_wide_spread:
                return 'wide_spread'
            elif is_low_liquidity:
                return 'illiquid'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"시장 상태 분류 실패: {str(e)}")
            return 'unknown'
            
    def get_market_state(self) -> Dict[str, Any]:
        """
        시장 상태 조회
        
        Returns:
            Dict[str, Any]: 시장 상태 정보
        """
        try:
            current_metrics = self._calculate_current_metrics()
            
            return {
                'state': self.market_state,
                'metrics': current_metrics,
                'timestamp': self.metrics['timestamp'][-1] if self.metrics['timestamp'] else None
            }
            
        except Exception as e:
            logger.error(f"시장 상태 조회 실패: {str(e)}")
            return {
                'state': 'unknown',
                'metrics': {
                    'volatility': 0.0,
                    'spread': 0.0,
                    'liquidity': 0.0,
                    'volume': 0.0
                },
                'timestamp': None
            }
            
    def get_average_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        평균 메트릭 계산
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            Dict[str, float]: 평균 메트릭
        """
        try:
            if not self.metrics['volatility']:
                return {
                    'avg_volatility': 0.0,
                    'avg_spread': 0.0,
                    'avg_liquidity': 0.0,
                    'avg_volume': 0.0
                }
                
            # 시간 범위 필터링
            indices = self._get_time_range_indices(start_time, end_time)
            
            return {
                'avg_volatility': np.mean([self.metrics['volatility'][i] for i in indices]),
                'avg_spread': np.mean([self.metrics['spread'][i] for i in indices]),
                'avg_liquidity': np.mean([self.metrics['liquidity'][i] for i in indices]),
                'avg_volume': np.mean([self.metrics['volume'][i] for i in indices])
            }
            
        except Exception as e:
            logger.error(f"평균 메트릭 계산 실패: {str(e)}")
            return {
                'avg_volatility': 0.0,
                'avg_spread': 0.0,
                'avg_liquidity': 0.0,
                'avg_volume': 0.0
            }
            
    def _get_time_range_indices(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[int]:
        """
        시간 범위에 해당하는 인덱스 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            List[int]: 인덱스 목록
        """
        try:
            indices = range(len(self.metrics['timestamp']))
            
            if start_time:
                indices = [
                    i for i in indices
                    if self.metrics['timestamp'][i] >= start_time
                ]
                
            if end_time:
                indices = [
                    i for i in indices
                    if self.metrics['timestamp'][i] <= end_time
                ]
                
            return indices
            
        except Exception as e:
            logger.error(f"시간 범위 인덱스 조회 실패: {str(e)}")
            return [] 