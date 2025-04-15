"""
성능 메트릭 수집 모듈

이 모듈은 트레이딩 시스템의 성능 메트릭을 수집하고 분석하는 역할을 담당합니다.
주요 기능:
- 실행 성능 메트릭 수집
- 리스크 메트릭 수집
- 포트폴리오 성과 분석
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMetricsCollector:
    """성능 메트릭 수집기"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.monitor_task = None
        
        # 메트릭 설정
        self.window_size = config.get('window_size', 100)
        self.update_interval = config.get('update_interval', 1.0)  # 초
        
        # 메트릭 가중치
        self.weights = {
            'execution': config.get('execution_weight', 0.4),
            'risk': config.get('risk_weight', 0.3),
            'portfolio': config.get('portfolio_weight', 0.3)
        }
        
        # 메트릭 데이터
        self.execution_metrics = deque(maxlen=self.window_size)
        self.risk_metrics = deque(maxlen=self.window_size)
        self.portfolio_metrics = deque(maxlen=self.window_size)
        
        # 성과 데이터
        self.performance_history = deque(maxlen=self.window_size)
        self.trade_history = []
        
    async def initialize(self):
        """수집기 초기화"""
        try:
            self.is_running = True
            self.monitor_task = asyncio.create_task(self._monitor_performance())
            
            logger.info("성능 메트릭 수집기 초기화 완료")
            
        except Exception as e:
            logger.error(f"성능 메트릭 수집기 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """수집기 종료"""
        try:
            self.is_running = False
            
            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("성능 메트릭 수집기 종료 완료")
            
        except Exception as e:
            logger.error(f"성능 메트릭 수집기 종료 실패: {str(e)}")
            raise
            
    async def _monitor_performance(self):
        """성능 모니터링"""
        try:
            while self.is_running:
                try:
                    # 메트릭 업데이트
                    await self._update_metrics()
                    
                    # 성과 분석
                    await self._analyze_performance()
                    
                    await asyncio.sleep(self.update_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"성능 모니터링 중 오류 발생: {str(e)}")
                    await asyncio.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"성능 모니터링 실패: {str(e)}")
            
    async def _update_metrics(self):
        """메트릭 업데이트"""
        try:
            # 실행 메트릭 업데이트
            if self.execution_metrics:
                execution_score = self._calculate_execution_score()
                logger.info(f"실행 성능 점수: {execution_score:.2f}")
                
            # 리스크 메트릭 업데이트
            if self.risk_metrics:
                risk_score = self._calculate_risk_score()
                logger.info(f"리스크 점수: {risk_score:.2f}")
                
            # 포트폴리오 메트릭 업데이트
            if self.portfolio_metrics:
                portfolio_score = self._calculate_portfolio_score()
                logger.info(f"포트폴리오 성과 점수: {portfolio_score:.2f}")
                
        except Exception as e:
            logger.error(f"메트릭 업데이트 실패: {str(e)}")
            
    def _calculate_execution_score(self) -> float:
        """실행 성능 점수 계산"""
        try:
            if not self.execution_metrics:
                return 0.0
                
            # 평균 메트릭 계산
            metrics = {
                'latency': np.mean([m.get('latency', 0.0) for m in self.execution_metrics]),
                'fill_rate': np.mean([m.get('fill_rate', 0.0) for m in self.execution_metrics]),
                'slippage': np.mean([m.get('slippage', 0.0) for m in self.execution_metrics]),
                'cost': np.mean([m.get('cost', 0.0) for m in self.execution_metrics])
            }
            
            # 메트릭 정규화
            normalized_metrics = {
                'latency': self._normalize_latency(metrics['latency']),
                'fill_rate': metrics['fill_rate'],
                'slippage': self._normalize_slippage(metrics['slippage']),
                'cost': self._normalize_cost(metrics['cost'])
            }
            
            # 가중 평균 계산
            weights = {
                'latency': 0.3,
                'fill_rate': 0.3,
                'slippage': 0.2,
                'cost': 0.2
            }
            
            score = sum(
                normalized_metrics[key] * weight
                for key, weight in weights.items()
            )
            
            return score
            
        except Exception as e:
            logger.error(f"실행 성능 점수 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_risk_score(self) -> float:
        """리스크 점수 계산"""
        try:
            if not self.risk_metrics:
                return 0.0
                
            # 평균 메트릭 계산
            metrics = {
                'position_risk': np.mean([m.get('position_risk', 0.0) for m in self.risk_metrics]),
                'volatility_risk': np.mean([m.get('volatility_risk', 0.0) for m in self.risk_metrics]),
                'liquidity_risk': np.mean([m.get('liquidity_risk', 0.0) for m in self.risk_metrics]),
                'concentration_risk': np.mean([m.get('concentration_risk', 0.0) for m in self.risk_metrics])
            }
            
            # 메트릭 정규화
            normalized_metrics = {
                key: max(0.0, 1.0 - value)
                for key, value in metrics.items()
            }
            
            # 가중 평균 계산
            weights = {
                'position_risk': 0.3,
                'volatility_risk': 0.3,
                'liquidity_risk': 0.2,
                'concentration_risk': 0.2
            }
            
            score = sum(
                normalized_metrics[key] * weight
                for key, weight in weights.items()
            )
            
            return score
            
        except Exception as e:
            logger.error(f"리스크 점수 계산 실패: {str(e)}")
            return 0.0
            
    def _calculate_portfolio_score(self) -> float:
        """포트폴리오 성과 점수 계산"""
        try:
            if not self.portfolio_metrics:
                return 0.0
                
            # 평균 메트릭 계산
            metrics = {
                'return': np.mean([m.get('return', 0.0) for m in self.portfolio_metrics]),
                'sharpe_ratio': np.mean([m.get('sharpe_ratio', 0.0) for m in self.portfolio_metrics]),
                'sortino_ratio': np.mean([m.get('sortino_ratio', 0.0) for m in self.portfolio_metrics]),
                'max_drawdown': np.mean([m.get('max_drawdown', 0.0) for m in self.portfolio_metrics])
            }
            
            # 메트릭 정규화
            normalized_metrics = {
                'return': self._normalize_return(metrics['return']),
                'sharpe_ratio': self._normalize_sharpe(metrics['sharpe_ratio']),
                'sortino_ratio': self._normalize_sortino(metrics['sortino_ratio']),
                'max_drawdown': self._normalize_drawdown(metrics['max_drawdown'])
            }
            
            # 가중 평균 계산
            weights = {
                'return': 0.4,
                'sharpe_ratio': 0.3,
                'sortino_ratio': 0.2,
                'max_drawdown': 0.1
            }
            
            score = sum(
                normalized_metrics[key] * weight
                for key, weight in weights.items()
            )
            
            return score
            
        except Exception as e:
            logger.error(f"포트폴리오 성과 점수 계산 실패: {str(e)}")
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
            
    def _normalize_return(self, return_: float) -> float:
        """수익률 정규화"""
        try:
            max_return = self.config.get('max_return', 0.1)  # 10%
            return min(1.0, max(0.0, return_ / max_return))
            
        except Exception as e:
            logger.error(f"수익률 정규화 실패: {str(e)}")
            return 0.0
            
    def _normalize_sharpe(self, sharpe: float) -> float:
        """샤프 비율 정규화"""
        try:
            max_sharpe = self.config.get('max_sharpe', 3.0)
            return min(1.0, max(0.0, sharpe / max_sharpe))
            
        except Exception as e:
            logger.error(f"샤프 비율 정규화 실패: {str(e)}")
            return 0.0
            
    def _normalize_sortino(self, sortino: float) -> float:
        """소르티노 비율 정규화"""
        try:
            max_sortino = self.config.get('max_sortino', 3.0)
            return min(1.0, max(0.0, sortino / max_sortino))
            
        except Exception as e:
            logger.error(f"소르티노 비율 정규화 실패: {str(e)}")
            return 0.0
            
    def _normalize_drawdown(self, drawdown: float) -> float:
        """최대 낙폭 정규화"""
        try:
            max_drawdown = self.config.get('max_drawdown', 0.2)  # 20%
            return max(0.0, 1.0 - (drawdown / max_drawdown))
            
        except Exception as e:
            logger.error(f"최대 낙폭 정규화 실패: {str(e)}")
            return 0.0
            
    async def _analyze_performance(self):
        """성과 분석"""
        try:
            # 전체 성과 점수 계산
            execution_score = self._calculate_execution_score()
            risk_score = self._calculate_risk_score()
            portfolio_score = self._calculate_portfolio_score()
            
            total_score = (
                execution_score * self.weights['execution'] +
                risk_score * self.weights['risk'] +
                portfolio_score * self.weights['portfolio']
            )
            
            # 성과 기록 저장
            self.performance_history.append({
                'timestamp': datetime.now(),
                'execution_score': execution_score,
                'risk_score': risk_score,
                'portfolio_score': portfolio_score,
                'total_score': total_score
            })
            
            logger.info(f"전체 성과 점수: {total_score:.2f}")
            
        except Exception as e:
            logger.error(f"성과 분석 실패: {str(e)}")
            
    async def add_execution_metrics(self, execution: Dict[str, Any]):
        """실행 메트릭 추가"""
        try:
            metrics = {
                'latency': execution.get('latency', 0.0),
                'fill_rate': execution.get('fill_rate', 0.0),
                'slippage': execution.get('slippage', 0.0),
                'cost': execution.get('cost', 0.0),
                'timestamp': datetime.now()
            }
            
            self.execution_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"실행 메트릭 추가 실패: {str(e)}")
            
    async def add_risk_metrics(self, risk_metrics: Dict[str, Any]):
        """리스크 메트릭 추가"""
        try:
            metrics = {
                'position_risk': risk_metrics.get('position_risk', 0.0),
                'volatility_risk': risk_metrics.get('volatility_risk', 0.0),
                'liquidity_risk': risk_metrics.get('liquidity_risk', 0.0),
                'concentration_risk': risk_metrics.get('concentration_risk', 0.0),
                'timestamp': datetime.now()
            }
            
            self.risk_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"리스크 메트릭 추가 실패: {str(e)}")
            
    async def add_portfolio_metrics(self, portfolio_metrics: Dict[str, Any]):
        """포트폴리오 메트릭 추가"""
        try:
            metrics = {
                'return': portfolio_metrics.get('return', 0.0),
                'sharpe_ratio': portfolio_metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': portfolio_metrics.get('sortino_ratio', 0.0),
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0.0),
                'timestamp': datetime.now()
            }
            
            self.portfolio_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"포트폴리오 메트릭 추가 실패: {str(e)}")
            
    async def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        try:
            return {
                'execution': {
                    'current': self.execution_metrics[-1] if self.execution_metrics else {},
                    'average': self._calculate_average_metrics(self.execution_metrics)
                },
                'risk': {
                    'current': self.risk_metrics[-1] if self.risk_metrics else {},
                    'average': self._calculate_average_metrics(self.risk_metrics)
                },
                'portfolio': {
                    'current': self.portfolio_metrics[-1] if self.portfolio_metrics else {},
                    'average': self._calculate_average_metrics(self.portfolio_metrics)
                },
                'performance': {
                    'current': self.performance_history[-1] if self.performance_history else {},
                    'history': list(self.performance_history)
                }
            }
            
        except Exception as e:
            logger.error(f"메트릭 조회 실패: {str(e)}")
            return {}
            
    def _calculate_average_metrics(self, metrics: deque) -> Dict[str, float]:
        """평균 메트릭 계산"""
        try:
            if not metrics:
                return {}
                
            result = {}
            for key in metrics[0].keys():
                if key != 'timestamp':
                    values = [m.get(key, 0.0) for m in metrics]
                    result[key] = sum(values) / len(values)
                    
            return result
            
        except Exception as e:
            logger.error(f"평균 메트릭 계산 실패: {str(e)}")
            return {}
