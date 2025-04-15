"""
실시간 시장 상태 분석기

이 모듈은 실시간으로 시장 상태를 분석하고 실행 전략을 조정하는 기능을 제공합니다.
주요 기능:
- 시장 변동성 분석
- 유동성 스코어 계산
- 거래량 불균형 감지
- 시장 충격 예측
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class MarketStateAnalyzer:
    """시장 상태 분석기"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        시장 상태 분석기 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        
        # 분석 설정
        self.update_interval = config.get('update_interval', 1)  # 초
        self.window_size = config.get('window_size', 300)  # 5분
        self.history_size = config.get('history_size', 1000)
        
        # 임계값 설정
        self.thresholds = {
            'volatility': config.get('volatility_threshold', 0.002),  # 0.2%
            'spread': config.get('spread_threshold', 0.001),  # 0.1%
            'volume_imbalance': config.get('volume_imbalance_threshold', 0.7),  # 70%
            'liquidity_score': config.get('liquidity_threshold', 0.5)  # 50%
        }
        
        # 상태 저장소
        self.market_states = []
        self.current_state = {
            'volatility': 0.0,
            'spread': 0.0,
            'volume_imbalance': 0.0,
            'liquidity_score': 1.0,
            'market_impact': 0.0,
            'timestamp': datetime.now()
        }
        
        # 분석 상태
        self.is_analyzing = False
        self.analysis_task = None
        
    async def initialize(self):
        """시장 상태 분석기 초기화"""
        try:
            # 분석 시작
            self.is_analyzing = True
            self.analysis_task = asyncio.create_task(self._analyze_market())
            
            logger.info("시장 상태 분석기 초기화 완료")
            
        except Exception as e:
            logger.error(f"시장 상태 분석기 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            self.is_analyzing = False
            if self.analysis_task:
                await self.analysis_task
                
            logger.info("시장 상태 분석기 종료")
            
        except Exception as e:
            logger.error(f"시장 상태 분석기 종료 실패: {str(e)}")
            
    async def _analyze_market(self):
        """시장 상태 분석 루프"""
        try:
            while self.is_analyzing:
                # 시장 상태 분석
                await self._update_market_state()
                
                # 이력 저장
                self.market_states.append(self.current_state.copy())
                
                # 이력 크기 제한
                if len(self.market_states) > self.history_size:
                    self.market_states = self.market_states[-self.history_size:]
                    
                # 대기
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"시장 상태 분석 실패: {str(e)}")
            
    async def _update_market_state(self):
        """시장 상태 업데이트"""
        try:
            # 현재 시간
            now = datetime.now()
            
            # 시장 상태 업데이트
            self.current_state.update({
                'timestamp': now
            })
            
        except Exception as e:
            logger.error(f"시장 상태 업데이트 실패: {str(e)}")
            
    def update_market_metrics(
        self,
        price: float,
        volume: float,
        bid_price: float,
        ask_price: float,
        bid_volume: float,
        ask_volume: float
    ):
        """
        시장 메트릭 업데이트
        
        Args:
            price (float): 현재가
            volume (float): 거래량
            bid_price (float): 매수 호가
            ask_price (float): 매도 호가
            bid_volume (float): 매수 잔량
            ask_volume (float): 매도 잔량
        """
        try:
            # 스프레드 계산
            spread = (ask_price - bid_price) / price
            
            # 거래량 불균형 계산
            total_volume = bid_volume + ask_volume
            volume_imbalance = abs(bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # 변동성 계산 (최근 N개 가격 기준)
            if len(self.market_states) > 0:
                prices = [state.get('price', price) for state in self.market_states[-self.window_size:]]
                prices.append(price)
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) if len(returns) > 0 else 0
            else:
                volatility = 0
                
            # 유동성 스코어 계산
            liquidity_score = self._calculate_liquidity_score(
                spread,
                volume,
                total_volume
            )
            
            # 시장 충격 예측
            market_impact = self._estimate_market_impact(
                volume,
                total_volume,
                spread
            )
            
            # 상태 업데이트
            self.current_state.update({
                'price': price,
                'volume': volume,
                'spread': spread,
                'volume_imbalance': volume_imbalance,
                'volatility': volatility,
                'liquidity_score': liquidity_score,
                'market_impact': market_impact
            })
            
        except Exception as e:
            logger.error(f"시장 메트릭 업데이트 실패: {str(e)}")
            
    def _calculate_liquidity_score(
        self,
        spread: float,
        volume: float,
        total_volume: float
    ) -> float:
        """
        유동성 스코어 계산
        
        Args:
            spread (float): 스프레드
            volume (float): 거래량
            total_volume (float): 총 잔량
            
        Returns:
            float: 유동성 스코어 (0.0 ~ 1.0)
        """
        try:
            # 스프레드 점수 (좁을수록 높음)
            spread_score = max(0, 1 - spread / self.thresholds['spread'])
            
            # 거래량 점수 (많을수록 높음)
            volume_score = min(1, volume / (total_volume * 0.1))
            
            # 잔량 점수 (많을수록 높음)
            depth_score = min(1, total_volume / (volume * 10))
            
            # 가중 평균 계산
            weights = {
                'spread': 0.4,
                'volume': 0.3,
                'depth': 0.3
            }
            
            score = (
                spread_score * weights['spread'] +
                volume_score * weights['volume'] +
                depth_score * weights['depth']
            )
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"유동성 스코어 계산 실패: {str(e)}")
            return 0.5
            
    def _estimate_market_impact(
        self,
        volume: float,
        total_volume: float,
        spread: float
    ) -> float:
        """
        시장 충격 예측
        
        Args:
            volume (float): 거래량
            total_volume (float): 총 잔량
            spread (float): 스프레드
            
        Returns:
            float: 예상 시장 충격 (가격 변화율)
        """
        try:
            # 거래량 비율
            volume_ratio = volume / total_volume if total_volume > 0 else 0
            
            # 기본 충격 = 거래량 비율 * 스프레드
            base_impact = volume_ratio * spread
            
            # 비선형 조정 (거래량이 많을수록 충격이 더 커짐)
            if volume_ratio > 0.1:  # 10% 이상
                base_impact *= (1 + np.log10(volume_ratio * 10))
                
            return base_impact
            
        except Exception as e:
            logger.error(f"시장 충격 예측 실패: {str(e)}")
            return 0.0
            
    def get_current_state(self) -> Dict[str, Any]:
        """
        현재 시장 상태 조회
        
        Returns:
            Dict[str, Any]: 현재 시장 상태
        """
        return self.current_state.copy()
        
    def get_state_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        시장 상태 이력 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            List[Dict[str, Any]]: 시장 상태 이력
        """
        try:
            # 시간 범위 필터링
            filtered_history = self.market_states
            
            if start_time:
                filtered_history = [
                    s for s in filtered_history
                    if s['timestamp'] >= start_time
                ]
                
            if end_time:
                filtered_history = [
                    s for s in filtered_history
                    if s['timestamp'] <= end_time
                ]
                
            return filtered_history
            
        except Exception as e:
            logger.error(f"시장 상태 이력 조회 실패: {str(e)}")
            return []
            
    def is_market_stable(self) -> bool:
        """
        시장 안정성 확인
        
        Returns:
            bool: 시장 안정 여부
        """
        try:
            return (
                self.current_state['volatility'] <= self.thresholds['volatility'] and
                self.current_state['spread'] <= self.thresholds['spread'] and
                self.current_state['volume_imbalance'] <= self.thresholds['volume_imbalance'] and
                self.current_state['liquidity_score'] >= self.thresholds['liquidity_score']
            )
            
        except Exception as e:
            logger.error(f"시장 안정성 확인 실패: {str(e)}")
            return False
            
    def get_execution_urgency(self) -> float:
        """
        실행 긴급도 계산
        
        Returns:
            float: 실행 긴급도 (0.0 ~ 1.0)
        """
        try:
            # 시장 상태 기반 긴급도 계산
            urgency_factors = {
                'volatility': max(0, self.current_state['volatility'] / self.thresholds['volatility']),
                'spread': max(0, 1 - self.current_state['spread'] / self.thresholds['spread']),
                'liquidity': self.current_state['liquidity_score']
            }
            
            # 가중 평균 계산
            weights = {
                'volatility': 0.4,
                'spread': 0.3,
                'liquidity': 0.3
            }
            
            urgency = sum(
                score * weights[factor]
                for factor, score in urgency_factors.items()
            )
            
            return max(0, min(1, urgency))
            
        except Exception as e:
            logger.error(f"실행 긴급도 계산 실패: {str(e)}")
            return 0.5 