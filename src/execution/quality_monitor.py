"""
실행 품질 모니터링 시스템

이 모듈은 주문 실행의 품질을 모니터링하고 분석합니다.
주요 기능:
- 실행 품질 지표 수집
- 품질 이슈 감지
- 품질 개선 제안
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class ExecutionQualityMonitor:
    """실행 품질 모니터"""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 모니터링 설정
        self.update_interval = config.get('update_interval', 1.0)  # 기본 1초
        self.window_size = config.get('window_size', 100)  # 기본 100개 샘플
        self.quality_threshold = config.get('quality_threshold', 0.9)  # 기본 90%
        
        # 품질 지표 가중치
        self.weights = {
            'latency': config.get('latency_weight', 0.3),
            'fill_rate': config.get('fill_rate_weight', 0.3),
            'slippage': config.get('slippage_weight', 0.2),
            'cost': config.get('cost_weight', 0.2)
        }
        
        # 품질 지표 저장소
        self.quality_history = deque(maxlen=self.window_size)
        self.issue_history = deque(maxlen=self.window_size)
        
        # 현재 상태
        self.current_quality = 0.0
        self.current_issues = []
        
        self.logger.info("실행 품질 모니터 초기화 완료")
        
    def update_quality(self, execution_data: Dict[str, Any]) -> None:
        """실행 품질 업데이트"""
        try:
            # 개별 품질 지표 계산
            metrics = {
                'latency': self._calculate_latency_quality(execution_data),
                'fill_rate': self._calculate_fill_rate_quality(execution_data),
                'slippage': self._calculate_slippage_quality(execution_data),
                'cost': self._calculate_cost_quality(execution_data)
            }
            
            # 종합 품질 점수 계산
            quality_score = sum(
                score * self.weights[metric]
                for metric, score in metrics.items()
            )
            
            # 품질 이슈 감지
            issues = self._detect_quality_issues(metrics)
            
            # 상태 업데이트
            self.quality_history.append(quality_score)
            self.current_quality = quality_score
            self.issue_history.append(issues)
            self.current_issues = issues
            
            self.logger.debug(f"실행 품질 업데이트: score={quality_score}, issues={issues}")
            
        except Exception as e:
            self.logger.error(f"실행 품질 업데이트 실패: {str(e)}")
            raise
            
    def get_quality_metrics(self) -> Dict[str, Any]:
        """품질 메트릭 조회"""
        try:
            return {
                'quality': {
                    'current': self.current_quality,
                    'average': np.mean(self.quality_history) if self.quality_history else 0.0,
                    'min': min(self.quality_history) if self.quality_history else 0.0,
                    'threshold': self.quality_threshold,
                    'is_healthy': self.current_quality >= self.quality_threshold
                },
                'issues': {
                    'current': self.current_issues,
                    'history': list(self.issue_history),
                    'count': len(self.current_issues)
                }
            }
            
        except Exception as e:
            self.logger.error(f"품질 메트릭 조회 실패: {str(e)}")
            raise
            
    def is_healthy(self) -> bool:
        """품질 상태 확인"""
        try:
            metrics = self.get_quality_metrics()
            return metrics['quality']['is_healthy'] and not metrics['issues']['current']
            
        except Exception as e:
            self.logger.error(f"품질 상태 확인 실패: {str(e)}")
            return False
            
    def get_improvement_suggestions(self) -> List[str]:
        """품질 개선 제안"""
        try:
            suggestions = []
            
            # 품질 이슈 기반 제안
            for issue in self.current_issues:
                if issue['type'] == 'latency':
                    suggestions.append("실행 지연시간을 줄이기 위해 주문 크기를 줄이거나 실행 전략을 조정하세요.")
                elif issue['type'] == 'fill_rate':
                    suggestions.append("체결률을 높이기 위해 주문 가격을 시장 가격에 더 가깝게 설정하세요.")
                elif issue['type'] == 'slippage':
                    suggestions.append("슬리피지를 줄이기 위해 주문을 더 작은 단위로 나누어 실행하세요.")
                elif issue['type'] == 'cost':
                    suggestions.append("실행 비용을 줄이기 위해 수수료가 낮은 거래소를 선택하거나 주문 유형을 변경하세요.")
                    
            # 품질 점수 기반 제안
            if self.current_quality < self.quality_threshold:
                suggestions.append("실행 품질을 개선하기 위해 실행 전략을 재검토하세요.")
                
            return suggestions
            
        except Exception as e:
            self.logger.error(f"품질 개선 제안 생성 실패: {str(e)}")
            return []
            
    def _calculate_latency_quality(self, execution_data: Dict[str, Any]) -> float:
        """지연시간 품질 계산"""
        try:
            latency = float(execution_data.get('latency', 0))
            threshold = self.config.get('latency_threshold', 1.0)
            
            if latency <= 0:
                return 1.0
                
            quality = 1.0 - (latency / threshold)
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"지연시간 품질 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_fill_rate_quality(self, execution_data: Dict[str, Any]) -> float:
        """체결률 품질 계산"""
        try:
            fill_rate = float(execution_data.get('fill_rate', 0))
            threshold = self.config.get('fill_rate_threshold', 0.95)
            
            if fill_rate >= threshold:
                return 1.0
                
            quality = fill_rate / threshold
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"체결률 품질 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_slippage_quality(self, execution_data: Dict[str, Any]) -> float:
        """슬리피지 품질 계산"""
        try:
            slippage = float(execution_data.get('slippage', 0))
            threshold = self.config.get('slippage_threshold', 0.001)
            
            if slippage <= 0:
                return 1.0
                
            quality = 1.0 - (slippage / threshold)
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"슬리피지 품질 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_cost_quality(self, execution_data: Dict[str, Any]) -> float:
        """실행 비용 품질 계산"""
        try:
            cost = float(execution_data.get('cost', 0))
            threshold = self.config.get('cost_threshold', 0.002)
            
            if cost <= 0:
                return 1.0
                
            quality = 1.0 - (cost / threshold)
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"실행 비용 품질 계산 실패: {str(e)}")
            return 0.0
            
    def _detect_quality_issues(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """품질 이슈 감지"""
        try:
            issues = []
            
            # 지연시간 이슈
            if metrics['latency'] < 0.8:  # 80% 미만
                issues.append({
                    'type': 'latency',
                    'severity': 'warning',
                    'message': '높은 실행 지연시간 감지'
                })
                
            # 체결률 이슈
            if metrics['fill_rate'] < 0.8:  # 80% 미만
                issues.append({
                    'type': 'fill_rate',
                    'severity': 'warning',
                    'message': '낮은 체결률 감지'
                })
                
            # 슬리피지 이슈
            if metrics['slippage'] < 0.8:  # 80% 미만
                issues.append({
                    'type': 'slippage',
                    'severity': 'warning',
                    'message': '높은 슬리피지 감지'
                })
                
            # 실행 비용 이슈
            if metrics['cost'] < 0.8:  # 80% 미만
                issues.append({
                    'type': 'cost',
                    'severity': 'warning',
                    'message': '높은 실행 비용 감지'
                })
                
            return issues
            
        except Exception as e:
            self.logger.error(f"품질 이슈 감지 실패: {str(e)}")
            return []

    async def initialize(self):
        """모니터 초기화"""
        try:
            self.is_running = True
            self.monitor_task = asyncio.create_task(self._monitor_quality())
            
            logger.info("실행 품질 모니터 초기화 완료")
            
        except Exception as e:
            logger.error(f"실행 품질 모니터 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """모니터 종료"""
        try:
            self.is_running = False
            
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("실행 품질 모니터 종료 완료")
            
        except Exception as e:
            logger.error(f"실행 품질 모니터 종료 실패: {str(e)}")
            raise
            
    async def _monitor_quality(self):
        """실행 품질 모니터링"""
        try:
            while self.is_running:
                try:
                    # 품질 메트릭 업데이트
                    await self._update_quality_metrics()
                    
                    # 비용 최적화
                    await self._optimize_costs()
                    
                    # 실행 전략 개선
                    await self._improve_strategy()
                    
                    await asyncio.sleep(self.update_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"실행 품질 모니터링 중 오류 발생: {str(e)}")
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"실행 품질 모니터링 실패: {str(e)}")
            
    async def _update_quality_metrics(self):
        """품질 메트릭 업데이트"""
        try:
            # 평균 메트릭 계산
            metrics = {}
            for key, values in self.quality_metrics.items():
                if values:
                    metrics[key] = sum(values) / len(values)
                else:
                    metrics[key] = 0.0
                    
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(metrics)
            
            # 품질 상태 로깅
            logger.info(f"실행 품질 점수: {quality_score:.2f}")
            logger.info(f"실행 메트릭: {metrics}")
            
        except Exception as e:
            logger.error(f"품질 메트릭 업데이트 실패: {str(e)}")
            
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """품질 점수 계산"""
        try:
            # 메트릭 정규화
            normalized_metrics = {
                'latency': self._normalize_latency(metrics['latency']),
                'fill_rate': metrics['fill_rate'],
                'slippage': self._normalize_slippage(metrics['slippage']),
                'cost': self._normalize_cost(metrics['cost'])
            }
            
            # 가중 평균 계산
            score = sum(
                normalized_metrics[key] * weight
                for key, weight in self.weights.items()
            )
            
            return score
            
        except Exception as e:
            logger.error(f"품질 점수 계산 실패: {str(e)}")
            return 0.0
            
    def _normalize_latency(self, latency: float) -> float:
        """지연시간 정규화"""
        try:
            max_latency = self.config.get('max_latency', 5.0)  # 초
            return max(0.0, 1.0 - (latency / max_latency))
            
        except Exception as e:
            logger.error(f"지연시간 정규화 실패: {str(e)}")
            return 0.0
            
    def _normalize_slippage(self, slippage: float) -> float:
        """슬리피지 정규화"""
        try:
            max_slippage = self.config.get('max_slippage', 0.01)
            return max(0.0, 1.0 - (slippage / max_slippage))
            
        except Exception as e:
            logger.error(f"슬리피지 정규화 실패: {str(e)}")
            return 0.0
            
    def _normalize_cost(self, cost: float) -> float:
        """비용 정규화"""
        try:
            max_cost = self.config.get('max_cost', 0.01)
            return max(0.0, 1.0 - (cost / max_cost))
            
        except Exception as e:
            logger.error(f"비용 정규화 실패: {str(e)}")
            return 0.0
            
    async def _optimize_costs(self):
        """비용 최적화"""
        try:
            # 현재 비용 분석
            current_cost = self._get_average_cost()
            
            if current_cost > self.cost_threshold:
                # 비용 최적화 파라미터 조정
                self._adjust_optimization_params()
                
                logger.info(f"비용 최적화 완료: {current_cost:.6f} -> {self._get_average_cost():.6f}")
                
        except Exception as e:
            logger.error(f"비용 최적화 실패: {str(e)}")
            
    def _get_average_cost(self) -> float:
        """평균 비용 계산"""
        try:
            if not self.quality_metrics['cost']:
                return 0.0
                
            return sum(self.quality_metrics['cost']) / len(self.quality_metrics['cost'])
            
        except Exception as e:
            logger.error(f"평균 비용 계산 실패: {str(e)}")
            return 0.0
            
    def _adjust_optimization_params(self):
        """최적화 파라미터 조정"""
        try:
            # 주문 크기 조정
            current_size = self.optimization_params['order_size']
            if current_size > 1.0:
                self.optimization_params['order_size'] = max(1.0, current_size * 0.9)
                
            # 시간 창 조정
            current_window = self.optimization_params['time_window']
            if current_window < 300:  # 5분
                self.optimization_params['time_window'] = min(300, current_window * 1.1)
                
            # 가격 개선 조정
            current_improvement = self.optimization_params['price_improvement']
            if current_improvement < 0.005:  # 0.5%
                self.optimization_params['price_improvement'] = min(0.005, current_improvement * 1.1)
                
        except Exception as e:
            logger.error(f"최적화 파라미터 조정 실패: {str(e)}")
            
    async def _improve_strategy(self):
        """실행 전략 개선"""
        try:
            # 품질 점수 확인
            quality_score = self._calculate_quality_score({
                key: sum(values) / len(values) if values else 0.0
                for key, values in self.quality_metrics.items()
            })
            
            if quality_score < self.quality_threshold:
                # 전략 개선 파라미터 조정
                self._adjust_strategy_params()
                
                logger.info(f"실행 전략 개선 완료: {quality_score:.2f}")
                
        except Exception as e:
            logger.error(f"실행 전략 개선 실패: {str(e)}")
            
    def _adjust_strategy_params(self):
        """전략 파라미터 조정"""
        try:
            # 주문 분할 전략 조정
            current_size = self.optimization_params['order_size']
            if current_size > 1.0:
                self.optimization_params['order_size'] = max(1.0, current_size * 0.8)
                
            # 시간 간격 조정
            current_window = self.optimization_params['time_window']
            if current_window > 60:  # 1분
                self.optimization_params['time_window'] = max(60, current_window * 0.9)
                
            # 가격 제한 조정
            current_improvement = self.optimization_params['price_improvement']
            if current_improvement > 0.001:  # 0.1%
                self.optimization_params['price_improvement'] = max(0.001, current_improvement * 0.9)
                
        except Exception as e:
            logger.error(f"전략 파라미터 조정 실패: {str(e)}")
            
    def add_execution(self, execution: Dict[str, Any]):
        """실행 추가"""
        try:
            # 실행 메트릭 추출
            metrics = {
                'latency': execution.get('latency', 0.0),
                'fill_rate': execution.get('fill_rate', 0.0),
                'slippage': execution.get('slippage', 0.0),
                'cost': execution.get('cost', 0.0)
            }
            
            # 메트릭 저장
            for key, value in metrics.items():
                self.quality_metrics[key].append(value)
                
            # 실행 기록 저장
            self.execution_history.append({
                'execution_id': execution.get('execution_id'),
                'order': execution.get('order'),
                'metrics': metrics,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"실행 추가 실패: {str(e)}")
            
    def get_quality_metrics(self) -> Dict[str, Any]:
        """품질 메트릭 조회"""
        try:
            metrics = {}
            for key, values in self.quality_metrics.items():
                if values:
                    metrics[key] = {
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
                else:
                    metrics[key] = {
                        'current': 0.0,
                        'average': 0.0,
                        'min': 0.0,
                        'max': 0.0
                    }
                    
            return metrics
            
        except Exception as e:
            logger.error(f"품질 메트릭 조회 실패: {str(e)}")
            return {}
            
    def get_optimization_params(self) -> Dict[str, Any]:
        """최적화 파라미터 조회"""
        return self.optimization_params.copy() 