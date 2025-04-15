"""
적응형 실행 알고리즘
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdaptiveExecutor:
    def __init__(self, config: Dict = None):
        """적응형 실행기 초기화"""
        self.config = config or {
            'learning_rate': 0.01,  # 학습률
            'exploration_rate': 0.1,  # 탐험률
            'min_trade_amount': 0.001,  # 최소 거래량
            'max_participation_rate': 0.2,  # 최대 참여율
            'price_tolerance': 0.001,  # 가격 허용 오차
            'volatility_window': 3600,  # 변동성 측정 윈도우 (초)
            'execution_interval': 10,  # 실행 간격 (초)
            'feature_window': 24,  # 특성 추출 윈도우 (시간)
            'reward_decay': 0.95  # 보상 감쇠율
        }
        
        self.active_executions = {}
        self.execution_results = {}
        self.market_states = {}
        self.strategy_weights = {}
        self.scaler = StandardScaler()
        
    async def execute_order(self, order: Dict, market_data: Dict) -> Dict:
        """주문 실행"""
        try:
            # 시장 상태 분석
            market_state = self._analyze_market_state(market_data)
            self._update_market_states(market_state)
            
            # 실행 전략 선택
            strategy = self._select_execution_strategy(market_state)
            
            # 실행 시작
            execution_id = self._generate_execution_id()
            self.active_executions[execution_id] = {
                'order': order,
                'strategy': strategy,
                'status': 'running',
                'start_time': datetime.now(),
                'market_state': market_state,
                'total_executed': 0.0,
                'total_cost': 0.0,
                'performance_metrics': {}
            }
            
            # 실행
            results = await self._execute_adaptive(execution_id, market_data)
            
            # 학습 및 전략 가중치 업데이트
            self._update_strategy_weights(execution_id, results)
            
            # 결과 저장
            self.execution_results[execution_id] = results
            
            return results
            
        except Exception as e:
            logger.error(f"적응형 실행 중 오류 발생: {str(e)}")
            raise
            
    def _analyze_market_state(self, market_data: Dict) -> np.ndarray:
        """시장 상태 분석"""
        try:
            # 특성 추출
            features = self._extract_features(market_data)
            
            # 특성 정규화
            if len(self.market_states) > 0:
                normalized_features = self.scaler.transform([features])[0]
            else:
                normalized_features = self.scaler.fit_transform([features])[0]
                
            return normalized_features
            
        except Exception as e:
            logger.error(f"시장 상태 분석 중 오류 발생: {str(e)}")
            return np.zeros(8)  # 기본 특성 수
            
    def _extract_features(self, market_data: Dict) -> np.ndarray:
        """특성 추출"""
        try:
            # 기본 특성
            volatility = self._estimate_volatility(market_data)
            spread = market_data.get('spread', 0.0)
            volume = market_data.get('volume', 0.0)
            depth = market_data.get('depth', 0.0)
            
            # 파생 특성
            volume_imbalance = self._calculate_volume_imbalance(market_data)
            price_trend = self._calculate_price_trend(market_data)
            market_impact = self._estimate_market_impact(volume, market_data)
            liquidity_score = self._calculate_liquidity_score(market_data)
            
            return np.array([
                volatility,
                spread,
                volume,
                depth,
                volume_imbalance,
                price_trend,
                market_impact,
                liquidity_score
            ])
            
        except Exception as e:
            logger.error(f"특성 추출 중 오류 발생: {str(e)}")
            return np.zeros(8)
            
    def _select_execution_strategy(self, market_state: np.ndarray) -> str:
        """실행 전략 선택"""
        try:
            # 전략 목록
            strategies = ['vwap', 'twap', 'is', 'pov']
            
            # 탐험 여부 결정
            if np.random.random() < self.config['exploration_rate']:
                return np.random.choice(strategies)
                
            # 전략별 가중치 계산
            weights = []
            for strategy in strategies:
                if strategy not in self.strategy_weights:
                    self.strategy_weights[strategy] = np.random.random(len(market_state))
                    
                weight = np.dot(market_state, self.strategy_weights[strategy])
                weights.append(weight)
                
            # 최적 전략 선택
            return strategies[np.argmax(weights)]
            
        except Exception as e:
            logger.error(f"전략 선택 중 오류 발생: {str(e)}")
            return 'twap'  # 기본 전략
            
    async def _execute_adaptive(self, execution_id: str, market_data: Dict) -> Dict:
        """적응형 실행"""
        try:
            execution = self.active_executions[execution_id]
            order = execution['order']
            strategy = execution['strategy']
            
            # 전략별 실행기 생성
            executor = self._create_strategy_executor(strategy)
            
            # 실행
            results = await executor.execute_order(order, market_data)
            
            # 성과 지표 계산
            performance_metrics = self._calculate_performance_metrics(
                execution_id,
                results,
                market_data
            )
            
            # 실행 정보 업데이트
            execution['performance_metrics'] = performance_metrics
            execution['status'] = 'completed'
            
            return {
                'execution_id': execution_id,
                'strategy': strategy,
                'results': results,
                'performance_metrics': performance_metrics,
                'completion_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"적응형 실행 중 오류 발생: {str(e)}")
            raise
            
    def _create_strategy_executor(self, strategy: str):
        """전략별 실행기 생성"""
        try:
            if strategy == 'vwap':
                from .vwap_executor import VWAPExecutor
                return VWAPExecutor(self.config)
            elif strategy == 'twap':
                from .twap_executor import TWAPExecutor
                return TWAPExecutor(self.config)
            elif strategy == 'is':
                from .is_executor import ISExecutor
                return ISExecutor(self.config)
            elif strategy == 'pov':
                from .pov_executor import POVExecutor
                return POVExecutor(self.config)
            else:
                raise ValueError(f"지원하지 않는 전략: {strategy}")
                
        except Exception as e:
            logger.error(f"전략 실행기 생성 중 오류 발생: {str(e)}")
            raise
            
    def _calculate_performance_metrics(
        self,
        execution_id: str,
        results: Dict,
        market_data: Dict
    ) -> Dict:
        """성과 지표 계산"""
        try:
            execution = self.active_executions[execution_id]
            
            # 실행 비용
            implementation_shortfall = results.get('implementation_shortfall', 0.0)
            
            # 시장 충격
            market_impact = self._estimate_market_impact(
                results.get('total_executed', 0.0),
                market_data
            )
            
            # 실행 속도
            execution_time = (
                results.get('completion_time', datetime.now()) -
                execution['start_time']
            ).total_seconds()
            
            # 목표 달성도
            target_amount = execution['order']['amount']
            executed_amount = results.get('total_executed', 0.0)
            completion_rate = executed_amount / target_amount if target_amount > 0 else 0.0
            
            return {
                'implementation_shortfall': implementation_shortfall,
                'market_impact': market_impact,
                'execution_time': execution_time,
                'completion_rate': completion_rate
            }
            
        except Exception as e:
            logger.error(f"성과 지표 계산 중 오류 발생: {str(e)}")
            return {}
            
    def _update_strategy_weights(self, execution_id: str, results: Dict):
        """전략 가중치 업데이트"""
        try:
            execution = self.active_executions[execution_id]
            strategy = execution['strategy']
            market_state = execution['market_state']
            metrics = execution['performance_metrics']
            
            # 보상 계산
            reward = self._calculate_reward(metrics)
            
            # 가중치 업데이트
            if strategy in self.strategy_weights:
                self.strategy_weights[strategy] += (
                    self.config['learning_rate'] *
                    reward *
                    market_state
                )
                
        except Exception as e:
            logger.error(f"전략 가중치 업데이트 중 오류 발생: {str(e)}")
            
    def _calculate_reward(self, metrics: Dict) -> float:
        """보상 계산"""
        try:
            # 각 지표별 가중치
            weights = {
                'implementation_shortfall': -1.0,  # 낮을수록 좋음
                'market_impact': -0.5,  # 낮을수록 좋음
                'execution_time': -0.3,  # 낮을수록 좋음
                'completion_rate': 1.0  # 높을수록 좋음
            }
            
            # 보상 계산
            reward = sum(
                weights.get(metric, 0.0) * value
                for metric, value in metrics.items()
            )
            
            return reward
            
        except Exception as e:
            logger.error(f"보상 계산 중 오류 발생: {str(e)}")
            return 0.0
            
    def _update_market_states(self, market_state: np.ndarray):
        """시장 상태 이력 업데이트"""
        try:
            timestamp = datetime.now()
            self.market_states[timestamp] = market_state
            
            # 오래된 상태 제거
            cutoff_time = timestamp - timedelta(
                hours=self.config['feature_window']
            )
            self.market_states = {
                t: s for t, s in self.market_states.items()
                if t >= cutoff_time
            }
            
        except Exception as e:
            logger.error(f"시장 상태 이력 업데이트 중 오류 발생: {str(e)}")
            
    def _estimate_volatility(self, market_data: Dict) -> float:
        """변동성 추정"""
        try:
            prices = market_data.get('historical_prices', [])
            if not prices:
                return 0.0
                
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(
                252 * 24 * 60 * 60 / self.config['volatility_window']
            )
            
            return volatility
            
        except Exception as e:
            logger.error(f"변동성 추정 중 오류 발생: {str(e)}")
            return 0.0
            
    def _calculate_volume_imbalance(self, market_data: Dict) -> float:
        """거래량 불균형 계산"""
        try:
            buy_volume = market_data.get('buy_volume', 0.0)
            sell_volume = market_data.get('sell_volume', 0.0)
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                return (buy_volume - sell_volume) / total_volume
            return 0.0
            
        except Exception as e:
            logger.error(f"거래량 불균형 계산 중 오류 발생: {str(e)}")
            return 0.0
            
    def _calculate_price_trend(self, market_data: Dict) -> float:
        """가격 추세 계산"""
        try:
            prices = market_data.get('historical_prices', [])
            if len(prices) < 2:
                return 0.0
                
            returns = np.diff(np.log(prices))
            trend = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            
            return trend
            
        except Exception as e:
            logger.error(f"가격 추세 계산 중 오류 발생: {str(e)}")
            return 0.0
            
    def _estimate_market_impact(self, amount: float, market_data: Dict) -> float:
        """시장 충격 추정"""
        try:
            daily_volume = market_data.get('daily_volume', 0.0)
            volatility = self._estimate_volatility(market_data)
            spread = market_data.get('spread', 0.0)
            
            if daily_volume > 0:
                participation_rate = amount / daily_volume
                impact = spread / 2 + volatility * np.sqrt(participation_rate)
                return impact
            return 0.0
            
        except Exception as e:
            logger.error(f"시장 충격 추정 중 오류 발생: {str(e)}")
            return 0.0
            
    def _calculate_liquidity_score(self, market_data: Dict) -> float:
        """유동성 점수 계산"""
        try:
            spread = market_data.get('spread', 0.0)
            depth = market_data.get('depth', 0.0)
            volume = market_data.get('volume', 0.0)
            
            if spread > 0:
                return (depth * volume) / spread
            return 0.0
            
        except Exception as e:
            logger.error(f"유동성 점수 계산 중 오류 발생: {str(e)}")
            return 0.0
            
    def _generate_execution_id(self) -> str:
        """실행 ID 생성"""
        import uuid
        return f"adaptive_{uuid.uuid4().hex[:8]}" 