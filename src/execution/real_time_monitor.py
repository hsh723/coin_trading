"""
실행 시스템 실시간 모니터링 모듈
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from src.execution.logger import ExecutionLogger
from src.execution.monitor import ExecutionMonitor
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """실시간 모니터"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        실시간 모니터 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        self.logger = ExecutionLogger(config)
        self.monitor = ExecutionMonitor(config)
        
        # 실시간 모니터링 설정
        self.update_interval = config['real_time'].get('update_interval', 1)  # 초
        self.window_size = config['real_time'].get('window_size', 60)  # 1분
        self.alert_thresholds = config['real_time'].get('alert_thresholds', {
            'spread': 0.001,  # 0.1%
            'volatility': 0.02,  # 2%
            'volume_imbalance': 0.7,  # 70%
            'market_impact': 0.001  # 0.1%
        })
        
        # 실시간 메트릭
        self.metrics = {
            'spread': [],
            'volatility': [],
            'volume_imbalance': [],
            'market_impact': [],
            'order_flow': [],
            'liquidity_score': []
        }
        
        # 시장 상태
        self.market_state = {
            'is_volatile': False,
            'is_liquid': True,
            'is_stable': True,
            'regime': 'normal'
        }
        
    async def initialize(self):
        """실시간 모니터 초기화"""
        try:
            # 로거 및 모니터 초기화
            await self.logger.initialize()
            await self.monitor.initialize()
            
            # 실시간 모니터링 시작
            asyncio.create_task(self._monitoring_loop())
            
            logger.info("실시간 모니터 초기화 완료")
            
        except Exception as e:
            logger.error(f"실시간 모니터 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            await self.logger.close()
            await self.monitor.close()
            logger.info("실시간 모니터 종료")
        except Exception as e:
            logger.error(f"실시간 모니터 종료 실패: {str(e)}")
            
    async def _monitoring_loop(self):
        """실시간 모니터링 루프"""
        try:
            while True:
                # 실시간 메트릭 수집
                await self._collect_real_time_metrics()
                
                # 시장 상태 분석
                await self._analyze_market_state()
                
                # 실행 품질 모니터링
                await self._monitor_execution_quality()
                
                # 대기
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"실시간 모니터링 루프 실행 중 오류 발생: {str(e)}")
            
    async def _collect_real_time_metrics(self):
        """실시간 메트릭 수집"""
        try:
            # 현재 시간
            now = datetime.now()
            start_time = now - timedelta(seconds=self.window_size)
            
            # 실행 로그 조회
            execution_logs = await self.logger.get_execution_logs(
                start_time=start_time,
                end_time=now
            )
            
            if len(execution_logs) > 0:
                # 스프레드 계산
                spreads = execution_logs['spread'].values
                self.metrics['spread'].append(np.mean(spreads))
                
                # 변동성 계산
                prices = execution_logs['price'].values
                returns = np.diff(np.log(prices)) if len(prices) > 1 else [0]
                self.metrics['volatility'].append(np.std(returns))
                
                # 거래량 불균형 계산
                buy_volume = execution_logs[execution_logs['side'] == 'buy']['volume'].sum()
                sell_volume = execution_logs[execution_logs['side'] == 'sell']['volume'].sum()
                total_volume = buy_volume + sell_volume
                if total_volume > 0:
                    imbalance = abs(buy_volume - sell_volume) / total_volume
                else:
                    imbalance = 0
                self.metrics['volume_imbalance'].append(imbalance)
                
                # 시장 충격 계산
                impacts = execution_logs['market_impact'].values
                self.metrics['market_impact'].append(np.mean(impacts))
                
                # 주문 흐름 계산
                order_flow = len(execution_logs)
                self.metrics['order_flow'].append(order_flow)
                
                # 유동성 점수 계산
                liquidity_score = self._calculate_liquidity_score(execution_logs)
                self.metrics['liquidity_score'].append(liquidity_score)
                
            # 메트릭 기록
            await self.logger.log_performance({
                'type': 'real_time',
                'metrics': {
                    name: values[-1] if len(values) > 0 else 0
                    for name, values in self.metrics.items()
                }
            })
            
        except Exception as e:
            logger.error(f"실시간 메트릭 수집 실패: {str(e)}")
            
    def _calculate_liquidity_score(self, logs: pd.DataFrame) -> float:
        """
        유동성 점수 계산
        
        Args:
            logs (pd.DataFrame): 실행 로그
            
        Returns:
            float: 유동성 점수 (0.0 ~ 1.0)
        """
        try:
            # 지표 가중치
            weights = {
                'spread': 0.3,
                'depth': 0.3,
                'volume': 0.2,
                'trades': 0.2
            }
            
            # 스프레드 점수 (낮을수록 좋음)
            spread_score = 1 - min(logs['spread'].mean() / 0.01, 1)
            
            # 호가창 깊이 점수
            depth_score = min(logs['depth'].mean() / 100, 1)
            
            # 거래량 점수
            volume_score = min(logs['volume'].sum() / 100, 1)
            
            # 거래 빈도 점수
            trades_score = min(len(logs) / 100, 1)
            
            # 종합 점수 계산
            liquidity_score = (
                weights['spread'] * spread_score +
                weights['depth'] * depth_score +
                weights['volume'] * volume_score +
                weights['trades'] * trades_score
            )
            
            return liquidity_score
            
        except Exception as e:
            logger.error(f"유동성 점수 계산 실패: {str(e)}")
            return 0.5
            
    async def _analyze_market_state(self):
        """시장 상태 분석"""
        try:
            if len(self.metrics['volatility']) > 0:
                # 변동성 상태 확인
                recent_volatility = self.metrics['volatility'][-1]
                self.market_state['is_volatile'] = (
                    recent_volatility > self.alert_thresholds['volatility']
                )
                
                # 유동성 상태 확인
                recent_liquidity = self.metrics['liquidity_score'][-1]
                self.market_state['is_liquid'] = recent_liquidity > 0.5
                
                # 안정성 상태 확인
                recent_impact = self.metrics['market_impact'][-1]
                self.market_state['is_stable'] = (
                    recent_impact < self.alert_thresholds['market_impact']
                )
                
                # 시장 레짐 판단
                self.market_state['regime'] = self._determine_market_regime()
                
                # 상태 기록
                await self.logger.log_performance({
                    'type': 'market_state',
                    'state': self.market_state
                })
                
        except Exception as e:
            logger.error(f"시장 상태 분석 실패: {str(e)}")
            
    def _determine_market_regime(self) -> str:
        """
        시장 레짐 판단
        
        Returns:
            str: 시장 레짐
        """
        try:
            if self.market_state['is_volatile']:
                if self.market_state['is_liquid']:
                    return 'volatile_liquid'
                else:
                    return 'volatile_illiquid'
            else:
                if self.market_state['is_liquid']:
                    return 'stable_liquid'
                else:
                    return 'stable_illiquid'
                    
        except Exception as e:
            logger.error(f"시장 레짐 판단 실패: {str(e)}")
            return 'normal'
            
    async def _monitor_execution_quality(self):
        """실행 품질 모니터링"""
        try:
            # 실행 품질 메트릭 계산
            quality_metrics = await self._calculate_execution_quality()
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(quality_metrics)
            
            # 품질 기록
            await self.logger.log_performance({
                'type': 'execution_quality',
                'metrics': quality_metrics,
                'score': quality_score
            })
            
        except Exception as e:
            logger.error(f"실행 품질 모니터링 실패: {str(e)}")
            
    async def _calculate_execution_quality(self) -> Dict[str, float]:
        """
        실행 품질 메트릭 계산
        
        Returns:
            Dict[str, float]: 실행 품질 메트릭
        """
        try:
            # 현재 시간
            now = datetime.now()
            start_time = now - timedelta(seconds=self.window_size)
            
            # 실행 로그 조회
            logs = await self.logger.get_execution_logs(
                start_time=start_time,
                end_time=now
            )
            
            if len(logs) > 0:
                return {
                    'fill_rate': logs['fill_rate'].mean(),
                    'price_improvement': logs['price_improvement'].mean(),
                    'timing_score': logs['timing_score'].mean(),
                    'cost_score': logs['cost_score'].mean()
                }
            else:
                return {
                    'fill_rate': 1.0,
                    'price_improvement': 0.0,
                    'timing_score': 0.5,
                    'cost_score': 0.5
                }
                
        except Exception as e:
            logger.error(f"실행 품질 메트릭 계산 실패: {str(e)}")
            return {
                'fill_rate': 1.0,
                'price_improvement': 0.0,
                'timing_score': 0.5,
                'cost_score': 0.5
            }
            
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        품질 점수 계산
        
        Args:
            metrics (Dict[str, float]): 실행 품질 메트릭
            
        Returns:
            float: 품질 점수 (0.0 ~ 1.0)
        """
        try:
            # 지표 가중치
            weights = {
                'fill_rate': 0.4,
                'price_improvement': 0.3,
                'timing_score': 0.2,
                'cost_score': 0.1
            }
            
            # 종합 점수 계산
            quality_score = sum(
                weight * metrics[name]
                for name, weight in weights.items()
            )
            
            return quality_score
            
        except Exception as e:
            logger.error(f"품질 점수 계산 실패: {str(e)}")
            return 0.5
            
    def get_market_state(self) -> Dict[str, Any]:
        """
        시장 상태 조회
        
        Returns:
            Dict[str, Any]: 시장 상태
        """
        return self.market_state.copy()
        
    async def get_real_time_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        실시간 메트릭 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 실시간 메트릭
        """
        try:
            # 성능 로그 조회
            logs = await self.logger.get_performance_logs(
                start_time=start_time,
                end_time=end_time
            )
            
            # 실시간 메트릭 필터링
            real_time_logs = logs[logs['type'] == 'real_time']
            
            # 데이터 변환
            metrics_data = []
            for _, row in real_time_logs.iterrows():
                metrics_data.append({
                    'timestamp': row['timestamp'],
                    **row['metrics']
                })
                
            return pd.DataFrame(metrics_data)
            
        except Exception as e:
            logger.error(f"실시간 메트릭 조회 실패: {str(e)}")
            return pd.DataFrame()
            
    async def get_execution_quality(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        실행 품질 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 실행 품질
        """
        try:
            # 성능 로그 조회
            logs = await self.logger.get_performance_logs(
                start_time=start_time,
                end_time=end_time
            )
            
            # 실행 품질 필터링
            quality_logs = logs[logs['type'] == 'execution_quality']
            
            # 데이터 변환
            quality_data = []
            for _, row in quality_logs.iterrows():
                quality_data.append({
                    'timestamp': row['timestamp'],
                    'score': row['score'],
                    **row['metrics']
                })
                
            return pd.DataFrame(quality_data)
            
        except Exception as e:
            logger.error(f"실행 품질 조회 실패: {str(e)}")
            return pd.DataFrame() 