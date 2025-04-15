"""
실행 시스템 성능 프로파일러
"""

import asyncio
import time
import cProfile
import pstats
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExecutionProfiler:
    """실행 시스템 성능 프로파일러"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        프로파일러 초기화
        
        Args:
            config (Dict[str, Any]): 설정
        """
        self.config = config
        self.profiler = cProfile.Profile()
        self.stats = defaultdict(list)
        self.start_time = None
        self.end_time = None
        
        # 성능 임계값 설정
        self.thresholds = {
            'execution_time': 1.0,  # 초
            'memory_usage': 100 * 1024 * 1024,  # 100MB
            'cpu_usage': 0.8,  # 80%
            'latency': 0.1  # 100ms
        }
        
    def start_profiling(self):
        """프로파일링 시작"""
        self.start_time = datetime.now()
        self.profiler.enable()
        logger.info("성능 프로파일링 시작")
        
    def stop_profiling(self):
        """프로파일링 종료"""
        self.profiler.disable()
        self.end_time = datetime.now()
        logger.info("성능 프로파일링 종료")
        
    def profile_execution(self, func):
        """
        실행 함수 프로파일링 데코레이터
        
        Args:
            func: 프로파일링할 함수
            
        Returns:
            프로파일링 래퍼 함수
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                # 함수 실행
                result = await func(*args, **kwargs)
                
                # 성능 지표 수집
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # 성능 통계 업데이트
                self.stats['execution_time'].append(execution_time)
                self.stats['memory_usage'].append(memory_delta)
                
                # 성능 경고 확인
                await self._check_performance_warnings({
                    'execution_time': execution_time,
                    'memory_usage': memory_delta,
                    'function': func.__name__
                })
                
                return result
                
            except Exception as e:
                logger.error(f"실행 프로파일링 중 오류 발생: {str(e)}")
                raise
                
        return wrapper
        
    async def _check_performance_warnings(self, metrics: Dict[str, Any]):
        """
        성능 경고 확인
        
        Args:
            metrics (Dict[str, Any]): 성능 지표
        """
        warnings = []
        
        # 실행 시간 확인
        if metrics['execution_time'] > self.thresholds['execution_time']:
            warnings.append(
                f"긴 실행 시간 감지: {metrics['function']} "
                f"({metrics['execution_time']:.2f}s)"
            )
            
        # 메모리 사용량 확인
        if metrics['memory_usage'] > self.thresholds['memory_usage']:
            warnings.append(
                f"높은 메모리 사용량 감지: {metrics['function']} "
                f"({metrics['memory_usage'] / 1024 / 1024:.2f}MB)"
            )
            
        # 경고 로깅
        for warning in warnings:
            logger.warning(warning)
            
    def _get_memory_usage(self) -> int:
        """
        현재 메모리 사용량 조회
        
        Returns:
            int: 메모리 사용량 (바이트)
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        성능 통계 조회
        
        Returns:
            Dict[str, Any]: 성능 통계
        """
        stats = {}
        
        for metric, values in self.stats.items():
            if values:
                stats[metric] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
                
        if self.start_time and self.end_time:
            stats['total_time'] = (self.end_time - self.start_time).total_seconds()
            
        return stats
        
    def generate_profile_report(self) -> Dict[str, Any]:
        """
        프로파일 보고서 생성
        
        Returns:
            Dict[str, Any]: 프로파일 보고서
        """
        # 프로파일러 통계 생성
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        # 상위 10개 함수 추출
        top_functions = []
        for func_stats in stats.stats.items():
            cc, nc, tt, ct, callers = func_stats[1]
            top_functions.append({
                'function': func_stats[0],
                'calls': cc,
                'time': tt,
                'cumulative_time': ct
            })
            if len(top_functions) >= 10:
                break
                
        # 성능 통계 조회
        performance_stats = self.get_performance_stats()
        
        # 보고서 생성
        report = {
            'timestamp': datetime.now(),
            'duration': performance_stats.get('total_time', 0),
            'top_functions': top_functions,
            'performance_stats': performance_stats,
            'warnings': [],
            'recommendations': []
        }
        
        # 성능 경고 추가
        for metric, values in performance_stats.items():
            if metric in self.thresholds:
                if values['max'] > self.thresholds[metric]:
                    report['warnings'].append(
                        f"높은 {metric} 감지: {values['max']}"
                    )
                    
        # 최적화 권장사항 추가
        self._add_optimization_recommendations(report)
        
        return report
        
    def _add_optimization_recommendations(self, report: Dict[str, Any]):
        """
        최적화 권장사항 추가
        
        Args:
            report (Dict[str, Any]): 프로파일 보고서
        """
        stats = report['performance_stats']
        
        # 실행 시간 최적화
        if 'execution_time' in stats:
            mean_time = stats['execution_time']['mean']
            if mean_time > self.thresholds['execution_time']:
                report['recommendations'].append(
                    "실행 시간이 긴 함수들의 알고리즘 최적화 필요"
                )
                
        # 메모리 사용량 최적화
        if 'memory_usage' in stats:
            mean_memory = stats['memory_usage']['mean']
            if mean_memory > self.thresholds['memory_usage']:
                report['recommendations'].append(
                    "메모리 사용량이 높은 함수들의 메모리 관리 개선 필요"
                )
                
        # 동시성 최적화
        concurrent_calls = len([
            f for f in report['top_functions']
            if f['calls'] > 1 and f['time'] > self.thresholds['execution_time']
        ])
        if concurrent_calls > 0:
            report['recommendations'].append(
                "자주 호출되는 느린 함수들의 비동기 처리 최적화 필요"
            )
            
    async def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        병목 지점 분석
        
        Returns:
            List[Dict[str, Any]]: 병목 지점 목록
        """
        bottlenecks = []
        stats = pstats.Stats(self.profiler)
        
        # 누적 시간 기준 정렬
        stats.sort_stats('cumulative')
        
        for func_stats in stats.stats.items():
            func_name = func_stats[0]
            cc, nc, tt, ct, callers = func_stats[1]
            
            # 병목 조건 확인
            is_bottleneck = (
                ct > self.thresholds['execution_time'] or
                cc > 100 or  # 과도한 호출
                tt / ct > 0.5  # 높은 자체 실행 시간 비율
            )
            
            if is_bottleneck:
                bottlenecks.append({
                    'function': func_name,
                    'calls': cc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'avg_time_per_call': tt / cc if cc > 0 else 0,
                    'callers': list(callers.keys()) if callers else []
                })
                
        return bottlenecks
        
    async def optimize_execution(self, bottlenecks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        실행 최적화 제안
        
        Args:
            bottlenecks (List[Dict[str, Any]]): 병목 지점 목록
            
        Returns:
            Dict[str, Any]: 최적화 제안
        """
        optimization_suggestions = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for bottleneck in bottlenecks:
            # 우선순위 결정
            if bottleneck['cumulative_time'] > self.thresholds['execution_time'] * 2:
                priority = 'high_priority'
            elif bottleneck['calls'] > 1000:
                priority = 'medium_priority'
            else:
                priority = 'low_priority'
                
            # 최적화 제안 생성
            suggestion = {
                'function': bottleneck['function'],
                'issue': self._analyze_performance_issue(bottleneck),
                'suggestions': self._generate_optimization_suggestions(bottleneck)
            }
            
            optimization_suggestions[priority].append(suggestion)
            
        return optimization_suggestions
        
    def _analyze_performance_issue(self, bottleneck: Dict[str, Any]) -> str:
        """
        성능 이슈 분석
        
        Args:
            bottleneck (Dict[str, Any]): 병목 정보
            
        Returns:
            str: 성능 이슈 설명
        """
        if bottleneck['avg_time_per_call'] > self.thresholds['execution_time']:
            return "높은 실행 시간"
        elif bottleneck['calls'] > 1000:
            return "과도한 함수 호출"
        elif len(bottleneck['callers']) > 10:
            return "높은 결합도"
        else:
            return "일반적인 성능 저하"
            
    def _generate_optimization_suggestions(self, bottleneck: Dict[str, Any]) -> List[str]:
        """
        최적화 제안 생성
        
        Args:
            bottleneck (Dict[str, Any]): 병목 정보
            
        Returns:
            List[str]: 최적화 제안 목록
        """
        suggestions = []
        
        # 실행 시간 최적화
        if bottleneck['avg_time_per_call'] > self.thresholds['execution_time']:
            suggestions.extend([
                "알고리즘 복잡도 개선",
                "캐싱 도입 검토",
                "비동기 처리 적용"
            ])
            
        # 호출 횟수 최적화
        if bottleneck['calls'] > 1000:
            suggestions.extend([
                "배치 처리 도입",
                "중복 호출 제거",
                "결과 캐싱"
            ])
            
        # 결합도 최적화
        if len(bottleneck['callers']) > 10:
            suggestions.extend([
                "인터페이스 단순화",
                "책임 분리",
                "이벤트 기반 구조 검토"
            ])
            
        return suggestions 