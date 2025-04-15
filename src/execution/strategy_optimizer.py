"""
실행 전략 최적화기

시장 상태와 과거 성능 데이터를 분석하여 최적의 실행 전략을 선택하는 모듈입니다.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExecutionStrategyOptimizer:
    """실행 전략 최적화기"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        실행 전략 최적화기 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        
        # 전략 목록
        self.strategies = config.get('strategies', [
            'twap', 'vwap', 'market', 'limit', 'iceberg', 'adaptive'
        ])
        
        # 시장 상태별 전략 가중치 (초기값)
        self.strategy_weights = {
            'normal': {
                'twap': 0.15,
                'vwap': 0.25,
                'market': 0.15,
                'limit': 0.15,
                'iceberg': 0.15,
                'adaptive': 0.15
            },
            'volatile': {
                'twap': 0.30,
                'vwap': 0.15,
                'market': 0.05,
                'limit': 0.15,
                'iceberg': 0.25,
                'adaptive': 0.10
            },
            'trending': {
                'twap': 0.15,
                'vwap': 0.30,
                'market': 0.20,
                'limit': 0.10,
                'iceberg': 0.10,
                'adaptive': 0.15
            },
            'illiquid': {
                'twap': 0.20,
                'vwap': 0.10,
                'market': 0.05,
                'limit': 0.20,
                'iceberg': 0.30,
                'adaptive': 0.15
            },
            'ranging': {
                'twap': 0.15,
                'vwap': 0.15,
                'market': 0.10,
                'limit': 0.30,
                'iceberg': 0.15,
                'adaptive': 0.15
            }
        }
        
        # 전략별 성능 메트릭 이력
        self.performance_history = defaultdict(lambda: defaultdict(list))
        
        # 최적화 설정
        self.optimization_interval = config.get('optimization_interval', 24)  # 시간
        self.min_samples = config.get('min_samples', 10)  # 최소 샘플 수
        self.weight_decay = config.get('weight_decay', 0.95)  # 가중치 감소 계수
        self.exploration_rate = config.get('exploration_rate', 0.1)  # 탐색 비율
        
        # 상태 변수
        self.last_optimization_time = datetime.now() - timedelta(hours=self.optimization_interval)
        self.is_running = False
        self.optimization_task = None
        
    async def initialize(self):
        """초기화"""
        try:
            # 과거 데이터 로드 (설정된 경우)
            if self.config.get('load_history', False):
                await self._load_performance_history()
                
            # 최적화 작업 시작
            self.is_running = True
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            logger.info("실행 전략 최적화기 초기화 완료")
            
        except Exception as e:
            logger.error(f"실행 전략 최적화기 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            self.is_running = False
            
            if self.optimization_task:
                self.optimization_task.cancel()
                try:
                    await self.optimization_task
                except asyncio.CancelledError:
                    pass
                
            # 성능 이력 저장 (설정된 경우)
            if self.config.get('save_history', False):
                await self._save_performance_history()
                
            logger.info("실행 전략 최적화기 종료")
            
        except Exception as e:
            logger.error(f"실행 전략 최적화기 종료 실패: {str(e)}")
            
    async def _optimization_loop(self):
        """최적화 루프"""
        try:
            while self.is_running:
                now = datetime.now()
                hours_since_last = (now - self.last_optimization_time).total_seconds() / 3600
                
                # 주기적으로 최적화 수행
                if hours_since_last >= self.optimization_interval:
                    await self._optimize_strategy_weights()
                    self.last_optimization_time = now
                    
                # 1시간 대기
                await asyncio.sleep(3600)
                
        except asyncio.CancelledError:
            logger.info("최적화 루프 취소됨")
        except Exception as e:
            logger.error(f"최적화 루프 실패: {str(e)}")
            
    async def _optimize_strategy_weights(self):
        """전략 가중치 최적화"""
        try:
            logger.info("전략 가중치 최적화 시작")
            
            # 각 시장 상태별로 최적화
            for market_state in self.strategy_weights.keys():
                # 충분한 데이터가 있는지 확인
                if not self._has_sufficient_data(market_state):
                    logger.info(f"{market_state} 상태에 대한 충분한 데이터가 없습니다")
                    continue
                    
                # 성능 데이터 분석
                strategy_scores = self._analyze_performance(market_state)
                
                # 가중치 업데이트
                self._update_weights(market_state, strategy_scores)
                
            logger.info("전략 가중치 최적화 완료")
            
        except Exception as e:
            logger.error(f"전략 가중치 최적화 실패: {str(e)}")
            
    def _has_sufficient_data(self, market_state: str) -> bool:
        """
        충분한 데이터가 있는지 확인
        
        Args:
            market_state (str): 시장 상태
            
        Returns:
            bool: 충분한 데이터 여부
        """
        for strategy in self.strategies:
            if len(self.performance_history[market_state][strategy]) < self.min_samples:
                return False
        return True
        
    def _analyze_performance(self, market_state: str) -> Dict[str, float]:
        """
        성능 데이터 분석
        
        Args:
            market_state (str): 시장 상태
            
        Returns:
            Dict[str, float]: 전략별 성능 점수
        """
        strategy_scores = {}
        
        for strategy in self.strategies:
            # 성능 이력 가져오기
            history = self.performance_history[market_state][strategy]
            
            if not history:
                strategy_scores[strategy] = 0.0
                continue
                
            # 최근 데이터에 더 큰 가중치 부여
            weighted_history = []
            weight = 1.0
            
            for score in reversed(history):  # 최근 데이터부터 처리
                weighted_history.append(score * weight)
                weight *= self.weight_decay
                
            # 가중 평균 계산
            total_weight = (1 - self.weight_decay ** len(history)) / (1 - self.weight_decay)
            weighted_avg = sum(weighted_history) / total_weight
            
            strategy_scores[strategy] = weighted_avg
            
        return strategy_scores
        
    def _update_weights(self, market_state: str, strategy_scores: Dict[str, float]):
        """
        가중치 업데이트
        
        Args:
            market_state (str): 시장 상태
            strategy_scores (Dict[str, float]): 전략별 성능 점수
        """
        # 점수 정규화
        total_score = sum(strategy_scores.values())
        
        if total_score <= 0:
            # 모든 점수가 0 이하인 경우 균등 분배
            for strategy in self.strategies:
                self.strategy_weights[market_state][strategy] = 1.0 / len(self.strategies)
            return
            
        # 탐색 비율 적용
        exploration = self.exploration_rate / len(self.strategies)
        exploitation = 1.0 - self.exploration_rate
        
        # 새 가중치 계산
        new_weights = {}
        for strategy in self.strategies:
            normalized_score = strategy_scores[strategy] / total_score
            new_weights[strategy] = exploration + (normalized_score * exploitation)
            
        # 정규화
        total_new_weight = sum(new_weights.values())
        for strategy in self.strategies:
            self.strategy_weights[market_state][strategy] = new_weights[strategy] / total_new_weight
            
        logger.info(f"{market_state} 상태의 전략 가중치 업데이트: {self.strategy_weights[market_state]}")
        
    def add_execution_result(
        self,
        strategy: str,
        market_state: str,
        performance_score: float
    ):
        """
        실행 결과 추가
        
        Args:
            strategy (str): 실행 전략
            market_state (str): 시장 상태
            performance_score (float): 성능 점수
        """
        try:
            # 유효한 전략 및 시장 상태 확인
            if strategy not in self.strategies:
                logger.warning(f"알 수 없는 전략: {strategy}")
                return
                
            if market_state not in self.strategy_weights:
                logger.warning(f"알 수 없는 시장 상태: {market_state}")
                return
                
            # 성능 이력에 추가
            self.performance_history[market_state][strategy].append(performance_score)
            
            # 이력 크기 제한
            max_history = self.config.get('max_history_per_strategy', 1000)
            if len(self.performance_history[market_state][strategy]) > max_history:
                self.performance_history[market_state][strategy] = self.performance_history[market_state][strategy][-max_history:]
                
            logger.debug(f"실행 결과 추가: {strategy}, {market_state}, {performance_score}")
            
        except Exception as e:
            logger.error(f"실행 결과 추가 실패: {str(e)}")
            
    def get_optimal_strategy(
        self,
        market_state: str,
        order_details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        최적 전략 선택
        
        Args:
            market_state (str): 시장 상태
            order_details (Optional[Dict[str, Any]]): 주문 상세 정보
            
        Returns:
            str: 선택된 전략
        """
        try:
            # 유효한 시장 상태 확인
            if market_state not in self.strategy_weights:
                logger.warning(f"알 수 없는 시장 상태: {market_state}, 기본값 'normal' 사용")
                market_state = 'normal'
                
            # 주문 상세 정보가 있는 경우 추가 고려
            if order_details:
                # 주문 특성에 따른 전략 조정
                adjusted_weights = self._adjust_weights_for_order(market_state, order_details)
            else:
                adjusted_weights = self.strategy_weights[market_state]
                
            # 전략 선택 (가중치 기준)
            strategies = list(adjusted_weights.keys())
            weights = list(adjusted_weights.values())
            
            chosen_strategy = np.random.choice(strategies, p=weights)
            
            logger.debug(f"선택된 전략: {chosen_strategy} (시장 상태: {market_state})")
            return chosen_strategy
            
        except Exception as e:
            logger.error(f"최적 전략 선택 실패: {str(e)}, 기본 전략 'vwap' 반환")
            return 'vwap'  # 기본 전략
            
    def _adjust_weights_for_order(
        self,
        market_state: str,
        order_details: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        주문 특성에 따른 가중치 조정
        
        Args:
            market_state (str): 시장 상태
            order_details (Dict[str, Any]): 주문 상세 정보
            
        Returns:
            Dict[str, float]: 조정된 가중치
        """
        try:
            # 기본 가중치 복사
            adjusted_weights = self.strategy_weights[market_state].copy()
            
            # 주문 크기에 따른 조정
            if 'quantity' in order_details and 'market_volume' in order_details:
                order_size = float(order_details['quantity'])
                market_volume = float(order_details['market_volume'])
                
                # 대형 주문 (시장 볼륨의 5% 이상)
                if market_volume > 0 and (order_size / market_volume) >= 0.05:
                    # 아이스버그 및 TWAP 전략 선호
                    self._boost_strategies(
                        adjusted_weights, ['iceberg', 'twap'], 1.5)
                    
            # 긴급 주문에 대한 조정
            if order_details.get('urgent', False):
                # 시장가 및 적응형 전략 선호
                self._boost_strategies(
                    adjusted_weights, ['market', 'adaptive'], 2.0)
                    
            # 스프레드에 따른 조정
            if 'spread' in order_details:
                spread = float(order_details['spread'])
                
                # 스프레드가 큰 경우
                if spread > 0.002:  # 0.2% 이상
                    # 지정가 및 TWAP 전략 선호
                    self._boost_strategies(
                        adjusted_weights, ['limit', 'twap'], 1.5)
                    
            # 가중치 정규화
            total_weight = sum(adjusted_weights.values())
            for strategy in adjusted_weights:
                adjusted_weights[strategy] /= total_weight
                
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"주문별 가중치 조정 실패: {str(e)}")
            return self.strategy_weights[market_state]
            
    def _boost_strategies(
        self,
        weights: Dict[str, float],
        strategies: List[str],
        boost_factor: float
    ):
        """
        특정 전략의 가중치 증가
        
        Args:
            weights (Dict[str, float]): 가중치 딕셔너리
            strategies (List[str]): 증가시킬 전략 목록
            boost_factor (float): 증가 계수
        """
        for strategy in strategies:
            if strategy in weights:
                weights[strategy] *= boost_factor
                
    def get_strategy_performance(
        self,
        market_state: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        전략 성능 통계 조회
        
        Args:
            market_state (Optional[str]): 시장 상태
            
        Returns:
            Dict[str, Dict[str, float]]: 전략 성능 통계
        """
        try:
            performance_stats = {}
            
            # 시장 상태 목록 결정
            states = [market_state] if market_state else self.strategy_weights.keys()
            
            for state in states:
                if state not in self.strategy_weights:
                    continue
                    
                performance_stats[state] = {}
                
                for strategy in self.strategies:
                    history = self.performance_history[state][strategy]
                    
                    if not history:
                        performance_stats[state][strategy] = {
                            'count': 0,
                            'avg': 0.0,
                            'max': 0.0,
                            'min': 0.0,
                            'weight': self.strategy_weights[state][strategy]
                        }
                        continue
                        
                    performance_stats[state][strategy] = {
                        'count': len(history),
                        'avg': np.mean(history),
                        'max': np.max(history),
                        'min': np.min(history),
                        'weight': self.strategy_weights[state][strategy]
                    }
                    
            return performance_stats
            
        except Exception as e:
            logger.error(f"전략 성능 통계 조회 실패: {str(e)}")
            return {}
            
    def get_strategy_weights(
        self,
        market_state: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        전략 가중치 조회
        
        Args:
            market_state (Optional[str]): 시장 상태
            
        Returns:
            Dict[str, Dict[str, float]]: 전략 가중치
        """
        try:
            if market_state:
                if market_state in self.strategy_weights:
                    return {market_state: self.strategy_weights[market_state]}
                return {}
            else:
                return self.strategy_weights
                
        except Exception as e:
            logger.error(f"전략 가중치 조회 실패: {str(e)}")
            return {}
            
    async def _load_performance_history(self):
        """성능 이력 로드"""
        try:
            # 파일 경로
            history_file = self.config.get('history_file', 'strategy_performance_history.json')
            
            # TODO: 구현 - 파일이나 데이터베이스에서 이력 로드
            logger.info(f"성능 이력 로드 (파일: {history_file})")
            
        except Exception as e:
            logger.error(f"성능 이력 로드 실패: {str(e)}")
            
    async def _save_performance_history(self):
        """성능 이력 저장"""
        try:
            # 파일 경로
            history_file = self.config.get('history_file', 'strategy_performance_history.json')
            
            # TODO: 구현 - 파일이나 데이터베이스에 이력 저장
            logger.info(f"성능 이력 저장 (파일: {history_file})")
            
        except Exception as e:
            logger.error(f"성능 이력 저장 실패: {str(e)}")
            
    def set_strategies(self, strategies: List[str]):
        """
        전략 목록 설정
        
        Args:
            strategies (List[str]): 전략 목록
        """
        self.strategies = strategies
        
        # 가중치 테이블 업데이트
        for market_state in self.strategy_weights:
            # 기존 가중치 유지
            current_weights = self.strategy_weights[market_state]
            
            # 새로운 가중치 테이블 생성
            new_weights = {}
            for strategy in strategies:
                if strategy in current_weights:
                    new_weights[strategy] = current_weights[strategy]
                else:
                    new_weights[strategy] = 1.0 / len(strategies)
                    
            # 가중치 정규화
            total_weight = sum(new_weights.values())
            for strategy in new_weights:
                new_weights[strategy] /= total_weight
                
            # 업데이트
            self.strategy_weights[market_state] = new_weights
            
        logger.info(f"전략 목록 업데이트: {strategies}")
        
    def set_market_states(self, market_states: List[str]):
        """
        시장 상태 목록 설정
        
        Args:
            market_states (List[str]): 시장 상태 목록
        """
        # 새로운 시장 상태 초기화
        for state in market_states:
            if state not in self.strategy_weights:
                # 초기 가중치 설정
                self.strategy_weights[state] = {
                    strategy: 1.0 / len(self.strategies)
                    for strategy in self.strategies
                }
                
        # 불필요한 시장 상태 제거
        for state in list(self.strategy_weights.keys()):
            if state not in market_states:
                del self.strategy_weights[state]
                
        logger.info(f"시장 상태 목록 업데이트: {market_states}")
        
    def reset_weights(self, market_state: Optional[str] = None):
        """
        가중치 초기화
        
        Args:
            market_state (Optional[str]): 초기화할 시장 상태 (None이면 모든 상태)
        """
        try:
            states = [market_state] if market_state else self.strategy_weights.keys()
            
            for state in states:
                if state in self.strategy_weights:
                    # 균등 가중치로 초기화
                    self.strategy_weights[state] = {
                        strategy: 1.0 / len(self.strategies)
                        for strategy in self.strategies
                    }
                    
            logger.info(f"가중치 초기화 완료: {states}")
            
        except Exception as e:
            logger.error(f"가중치 초기화 실패: {str(e)}")
            
    def manual_update_weight(
        self,
        market_state: str,
        strategy: str,
        weight: float
    ) -> bool:
        """
        가중치 수동 업데이트
        
        Args:
            market_state (str): 시장 상태
            strategy (str): 전략
            weight (float): 새 가중치
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 유효성 검사
            if market_state not in self.strategy_weights:
                logger.error(f"알 수 없는 시장 상태: {market_state}")
                return False
                
            if strategy not in self.strategies:
                logger.error(f"알 수 없는 전략: {strategy}")
                return False
                
            if weight < 0:
                logger.error(f"가중치는 음수일 수 없습니다: {weight}")
                return False
                
            # 가중치 업데이트
            self.strategy_weights[market_state][strategy] = weight
            
            # 나머지 가중치 조정
            remaining_weight = 1.0 - weight
            remaining_strategies = [s for s in self.strategies if s != strategy]
            
            if not remaining_strategies:
                return True
                
            # 기존 가중치 비율 유지하면서 조정
            total_other_weight = sum(self.strategy_weights[market_state][s] for s in remaining_strategies)
            
            if total_other_weight <= 0:
                # 균등 분배
                for s in remaining_strategies:
                    self.strategy_weights[market_state][s] = remaining_weight / len(remaining_strategies)
            else:
                # 비율 유지하면서 조정
                for s in remaining_strategies:
                    self.strategy_weights[market_state][s] = (
                        self.strategy_weights[market_state][s] / total_other_weight
                    ) * remaining_weight
                    
            logger.info(f"가중치 수동 업데이트: {market_state}, {strategy}, {weight}")
            return True
            
        except Exception as e:
            logger.error(f"가중치 수동 업데이트 실패: {str(e)}")
            return False 