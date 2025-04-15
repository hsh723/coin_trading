"""
실시간 실행 품질 모니터
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class ExecutionQualityMonitor:
    def __init__(self, config: dict):
        """
        실행 품질 모니터 초기화
        
        Args:
            config (dict): 설정
        """
        self.config = config
        self.update_interval = config.get('update_interval', 1)
        self.window_size = config.get('window_size', 100)
        
        self.thresholds = {
            'impact_threshold': config.get('impact_threshold', 0.001),
            'reversion_threshold': config.get('reversion_threshold', 0.002),
            'timing_score_threshold': config.get('timing_score_threshold', 0.7)
        }
        
        self.metrics = {
            'market_impact': [],
            'price_reversion': [],
            'timing_score': [],
            'execution_cost': [],
            'timestamp': [],
            'slippage': 0.0,
            'fill_rate': 1.0,
            'cost': 0.0
        }
        
        self.is_monitoring = False
        self.monitor_task = None
        
        self.quality_metrics = {
            'latency': {
                'value': 0.0,
                'weight': 0.3,
                'threshold': config.get('latency_threshold', 1.0)
            },
            'fill_rate': {
                'value': 0.0,
                'weight': 0.3,
                'threshold': config.get('fill_rate_threshold', 0.95)
            },
            'slippage': {
                'value': 0.0,
                'weight': 0.2,
                'threshold': config.get('slippage_threshold', 0.001)
            },
            'cost': {
                'value': 0.0,
                'weight': 0.2,
                'threshold': config.get('cost_threshold', 0.002)
            }
        }
        self.issues = []
        
    async def initialize(self):
        """모니터 초기화"""
        try:
            self.is_monitoring = True
            self.monitor_task = asyncio.create_task(self._monitor_quality())
            logger.info("실행 품질 모니터 초기화 완료")
        except Exception as e:
            logger.error(f"실행 품질 모니터 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """모니터 종료"""
        try:
            self.is_monitoring = False
            if self.monitor_task:
                await self.monitor_task
            logger.info("실행 품질 모니터 종료")
        except Exception as e:
            logger.error(f"실행 품질 모니터 종료 실패: {str(e)}")
            
    async def _monitor_quality(self):
        try:
            while self.is_monitoring:
                await self._update_metrics()
                await asyncio.sleep(self.update_interval)
        except Exception as e:
            logger.error(f"실행 품질 모니터링 실패: {str(e)}")
            
    async def _update_metrics(self):
        try:
            now = datetime.now()
            self._trim_old_metrics(now)
        except Exception as e:
            logger.error(f"메트릭 업데이트 실패: {str(e)}")
            
    def _trim_old_metrics(self, now: datetime):
        try:
            if len(self.metrics['timestamp']) > self.window_size:
                for key in self.metrics:
                    self.metrics[key] = self.metrics[key][-self.window_size:]
        except Exception as e:
            logger.error(f"메트릭 정리 실패: {str(e)}")
            
    def add_execution(
        self,
        execution_price: float,
        market_price: float,
        quantity: float,
        side: str,
        timestamp: datetime
    ):
        """
        실행 정보 추가
        
        Args:
            execution_price (float): 실행 가격
            market_price (float): 시장 가격
            quantity (float): 수량
            side (str): 매수/매도
            timestamp (datetime): 타임스탬프
        """
        try:
            # 시장 충격 계산
            impact = self._calculate_market_impact(
                execution_price,
                market_price,
                side
            )
            
            # 가격 반전 계산
            reversion = self._calculate_price_reversion(
                execution_price,
                market_price,
                side
            )
            
            # 타이밍 점수 계산
            timing = self._calculate_timing_score(
                execution_price,
                market_price,
                side
            )
            
            # 실행 비용 계산
            cost = self._calculate_execution_cost(
                execution_price,
                market_price,
                quantity,
                side
            )
            
            # 메트릭 저장
            self.metrics['market_impact'].append(impact)
            self.metrics['price_reversion'].append(reversion)
            self.metrics['timing_score'].append(timing)
            self.metrics['execution_cost'].append(cost)
            self.metrics['timestamp'].append(timestamp)
            
            logger.info(f"실행 품질 메트릭 추가: impact={impact:.6f}, reversion={reversion:.6f}, timing={timing:.2f}, cost={cost:.6f}")
            
            # 슬리피지 계산
            if side.upper() == 'BUY':
                slippage = (execution_price - market_price) / market_price
            else:
                slippage = (market_price - execution_price) / market_price
            
            self.metrics['slippage'] = slippage
            
        except Exception as e:
            logger.error(f"실행 메트릭 추가 실패: {str(e)}")
            
    def _calculate_market_impact(
        self,
        execution_price: float,
        market_price: float,
        side: str
    ) -> float:
        try:
            if side == 'buy':
                return max(0, (execution_price - market_price) / market_price)
            else:
                return max(0, (market_price - execution_price) / market_price)
        except Exception as e:
            logger.error(f"시장 충격 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_price_reversion(
        self,
        execution_price: float,
        market_price: float,
        side: str
    ) -> float:
        try:
            if side == 'buy':
                return max(0, (market_price - execution_price) / execution_price)
            else:
                return max(0, (execution_price - market_price) / execution_price)
        except Exception as e:
            logger.error(f"가격 반전 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_timing_score(
        self,
        execution_price: float,
        market_price: float,
        side: str
    ) -> float:
        try:
            price_diff = abs(execution_price - market_price) / market_price
            return max(0, min(1, 1 - price_diff / self.thresholds['impact_threshold']))
        except Exception as e:
            logger.error(f"타이밍 점수 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_execution_cost(
        self,
        execution_price: float,
        market_price: float,
        quantity: float,
        side: str
    ) -> float:
        try:
            if side == 'buy':
                return quantity * (execution_price - market_price)
            else:
                return quantity * (market_price - execution_price)
        except Exception as e:
            logger.error(f"실행 비용 계산 실패: {str(e)}")
            return 0.0
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        현재 메트릭 조회
        
        Returns:
            Dict[str, Any]: 품질 메트릭
        """
        try:
            if not any(self.metrics['market_impact']):
                return {
                    'market_impact': 0.0,
                    'price_reversion': 0.0,
                    'timing_score': 0.0,
                    'execution_cost': 0.0,
                    'slippage': 0.0,
                    'fill_rate': 1.0,
                    'cost': 0.0
                }
                
            return {
                'market_impact': self.metrics['market_impact'][-1],
                'price_reversion': self.metrics['price_reversion'][-1],
                'timing_score': self.metrics['timing_score'][-1],
                'execution_cost': self.metrics['execution_cost'][-1],
                'slippage': self.metrics['slippage'],
                'fill_rate': self.metrics['fill_rate'],
                'cost': self.metrics['cost'],
                'timestamp': self.metrics['timestamp'][-1]
            }
        except Exception as e:
            logger.error(f"현재 메트릭 조회 실패: {str(e)}")
            return {
                'market_impact': 0.0,
                'price_reversion': 0.0,
                'timing_score': 0.0,
                'execution_cost': 0.0,
                'slippage': 0.0,
                'fill_rate': 1.0,
                'cost': 0.0,
                'timestamp': datetime.now()
            }
            
    def get_average_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        try:
            if not any(self.metrics['market_impact']):
                return {
                    'avg_market_impact': 0.0,
                    'avg_price_reversion': 0.0,
                    'avg_timing_score': 0.0,
                    'avg_execution_cost': 0.0
                }
                
            # 시간 범위 필터링
            indices = self._get_time_range_indices(start_time, end_time)
            
            return {
                'avg_market_impact': np.mean([self.metrics['market_impact'][i] for i in indices]),
                'avg_price_reversion': np.mean([self.metrics['price_reversion'][i] for i in indices]),
                'avg_timing_score': np.mean([self.metrics['timing_score'][i] for i in indices]),
                'avg_execution_cost': np.mean([self.metrics['execution_cost'][i] for i in indices])
            }
        except Exception as e:
            logger.error(f"평균 메트릭 계산 실패: {str(e)}")
            return {
                'avg_market_impact': 0.0,
                'avg_price_reversion': 0.0,
                'avg_timing_score': 0.0,
                'avg_execution_cost': 0.0
            }
            
    def _get_time_range_indices(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> List[int]:
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
            
    def is_execution_quality_good(self) -> bool:
        """
        실행 품질 확인
        
        Returns:
            bool: 품질 양호 여부
        """
        try:
            current = self.get_current_metrics()
            return (
                current['market_impact'] <= self.thresholds['impact_threshold'] and
                current['price_reversion'] <= self.thresholds['reversion_threshold'] and
                current['timing_score'] >= self.thresholds['timing_score_threshold'] and
                current['slippage'] <= self.config.get('max_slippage', 0.002) and
                current['fill_rate'] >= self.config.get('min_fill_rate', 0.95) and
                current['cost'] <= self.config.get('max_cost', 0.001)
            )
        except Exception as e:
            logger.error(f"실행 품질 확인 실패: {str(e)}")
            return False

    def update_quality(self, metrics: dict):
        """품질 메트릭 업데이트"""
        try:
            for metric, data in metrics.items():
                if metric in self.quality_metrics:
                    self.quality_metrics[metric]['value'] = data['current']
            self._detect_issues()
        except Exception as e:
            logger.error(f"품질 메트릭 업데이트 실패: {str(e)}")

    def _detect_issues(self):
        """이슈 감지"""
        self.issues = []
        for metric, data in self.quality_metrics.items():
            if metric in ['latency', 'slippage', 'cost']:
                if data['value'] > data['threshold']:
                    self.issues.append({
                        'metric': metric,
                        'value': data['value'],
                        'threshold': data['threshold'],
                        'severity': 'high' if data['value'] > data['threshold'] * 1.5 else 'medium'
                    })
            elif metric == 'fill_rate':
                if data['value'] < data['threshold']:
                    self.issues.append({
                        'metric': metric,
                        'value': data['value'],
                        'threshold': data['threshold'],
                        'severity': 'high' if data['value'] < data['threshold'] * 0.5 else 'medium'
                    })

    def get_quality_score(self) -> float:
        """품질 점수 계산"""
        try:
            score = 0.0
            for metric, data in self.quality_metrics.items():
                if metric in ['latency', 'slippage', 'cost']:
                    normalized_value = max(0, 1 - (data['value'] / data['threshold']))
                else:  # fill_rate
                    normalized_value = min(1, data['value'] / data['threshold'])
                score += normalized_value * data['weight']
            return score
        except Exception as e:
            logger.error(f"품질 점수 계산 실패: {str(e)}")
            return 0.0

    def get_issues(self) -> list:
        """발견된 이슈 반환"""
        return self.issues

    def is_healthy(self) -> bool:
        """상태 확인"""
        return len(self.issues) == 0 