"""
실행 시스템 프로파일러 테스트
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, Any
from src.execution.profiler import ExecutionProfiler

@pytest.fixture
def profiler():
    """프로파일러 픽스처"""
    config = {
        'profiling': {
            'enabled': True,
            'thresholds': {
                'execution_time': 0.5,
                'memory_usage': 50 * 1024 * 1024,
                'cpu_usage': 0.7,
                'latency': 0.05
            }
        }
    }
    return ExecutionProfiler(config)

async def slow_function():
    """느린 테스트 함수"""
    await asyncio.sleep(0.6)
    return "완료"

async def memory_intensive_function():
    """메모리 사용이 많은 테스트 함수"""
    large_list = [i for i in range(1000000)]
    await asyncio.sleep(0.1)
    return len(large_list)

async def normal_function():
    """정상 테스트 함수"""
    await asyncio.sleep(0.1)
    return "정상"

@pytest.mark.asyncio
async def test_profile_execution(profiler):
    """실행 프로파일링 테스트"""
    # 프로파일링 데코레이터 적용
    slow_func = profiler.profile_execution(slow_function)
    memory_func = profiler.profile_execution(memory_intensive_function)
    normal_func = profiler.profile_execution(normal_function)
    
    # 프로파일링 시작
    profiler.start_profiling()
    
    # 함수 실행
    results = await asyncio.gather(
        slow_func(),
        memory_func(),
        normal_func()
    )
    
    # 프로파일링 종료
    profiler.stop_profiling()
    
    # 결과 검증
    assert results[0] == "완료"
    assert results[1] == 1000000
    assert results[2] == "정상"
    
    # 성능 통계 검증
    stats = profiler.get_performance_stats()
    assert 'execution_time' in stats
    assert 'memory_usage' in stats
    assert stats['execution_time']['count'] == 3
    
@pytest.mark.asyncio
async def test_performance_warnings(profiler):
    """성능 경고 테스트"""
    # 느린 함수 실행
    slow_func = profiler.profile_execution(slow_function)
    await slow_func()
    
    # 성능 통계 검증
    stats = profiler.get_performance_stats()
    assert stats['execution_time']['max'] > profiler.thresholds['execution_time']
    
@pytest.mark.asyncio
async def test_memory_usage_tracking(profiler):
    """메모리 사용량 추적 테스트"""
    # 메모리 사용이 많은 함수 실행
    memory_func = profiler.profile_execution(memory_intensive_function)
    await memory_func()
    
    # 메모리 통계 검증
    stats = profiler.get_performance_stats()
    assert 'memory_usage' in stats
    assert stats['memory_usage']['max'] > 0
    
@pytest.mark.asyncio
async def test_profile_report_generation(profiler):
    """프로파일 보고서 생성 테스트"""
    # 여러 함수 실행
    slow_func = profiler.profile_execution(slow_function)
    memory_func = profiler.profile_execution(memory_intensive_function)
    normal_func = profiler.profile_execution(normal_function)
    
    profiler.start_profiling()
    
    await asyncio.gather(
        slow_func(),
        memory_func(),
        normal_func()
    )
    
    profiler.stop_profiling()
    
    # 보고서 생성
    report = profiler.generate_profile_report()
    
    # 보고서 구조 검증
    assert 'timestamp' in report
    assert 'duration' in report
    assert 'top_functions' in report
    assert 'performance_stats' in report
    assert 'warnings' in report
    assert 'recommendations' in report
    
    # 성능 경고 검증
    assert len(report['warnings']) > 0
    assert any('execution_time' in warning for warning in report['warnings'])
    
@pytest.mark.asyncio
async def test_bottleneck_analysis(profiler):
    """병목 지점 분석 테스트"""
    # 여러 번 함수 실행
    slow_func = profiler.profile_execution(slow_function)
    
    profiler.start_profiling()
    
    for _ in range(5):
        await slow_func()
        
    profiler.stop_profiling()
    
    # 병목 지점 분석
    bottlenecks = await profiler.analyze_bottlenecks()
    
    # 분석 결과 검증
    assert len(bottlenecks) > 0
    assert bottlenecks[0]['function'].endswith('slow_function')
    assert bottlenecks[0]['calls'] == 5
    assert bottlenecks[0]['avg_time_per_call'] > profiler.thresholds['execution_time']
    
@pytest.mark.asyncio
async def test_optimization_suggestions(profiler):
    """최적화 제안 테스트"""
    # 여러 함수 실행
    slow_func = profiler.profile_execution(slow_function)
    memory_func = profiler.profile_execution(memory_intensive_function)
    
    profiler.start_profiling()
    
    for _ in range(5):
        await asyncio.gather(
            slow_func(),
            memory_func()
        )
        
    profiler.stop_profiling()
    
    # 병목 지점 분석
    bottlenecks = await profiler.analyze_bottlenecks()
    
    # 최적화 제안 생성
    suggestions = await profiler.optimize_execution(bottlenecks)
    
    # 제안 구조 검증
    assert 'high_priority' in suggestions
    assert 'medium_priority' in suggestions
    assert 'low_priority' in suggestions
    
    # 우선순위별 제안 검증
    high_priority = suggestions['high_priority']
    assert len(high_priority) > 0
    assert 'function' in high_priority[0]
    assert 'issue' in high_priority[0]
    assert 'suggestions' in high_priority[0]
    assert len(high_priority[0]['suggestions']) > 0
    
@pytest.mark.asyncio
async def test_concurrent_profiling(profiler):
    """동시 실행 프로파일링 테스트"""
    # 여러 함수 동시 실행
    async def concurrent_workload():
        tasks = []
        for _ in range(3):
            slow_func = profiler.profile_execution(slow_function)
            memory_func = profiler.profile_execution(memory_intensive_function)
            normal_func = profiler.profile_execution(normal_function)
            
            tasks.extend([
                slow_func(),
                memory_func(),
                normal_func()
            ])
            
        return await asyncio.gather(*tasks)
        
    profiler.start_profiling()
    results = await concurrent_workload()
    profiler.stop_profiling()
    
    # 결과 검증
    assert len(results) == 9  # 3개 함수 * 3회 반복
    
    # 성능 통계 검증
    stats = profiler.get_performance_stats()
    assert stats['execution_time']['count'] == 9
    
    # 보고서 생성 및 검증
    report = profiler.generate_profile_report()
    assert len(report['top_functions']) > 0
    assert len(report['recommendations']) > 0
    
@pytest.mark.asyncio
async def test_error_handling(profiler):
    """오류 처리 테스트"""
    # 오류 발생 함수
    @profiler.profile_execution
    async def error_function():
        raise ValueError("테스트 오류")
        
    profiler.start_profiling()
    
    # 오류 발생 확인
    with pytest.raises(ValueError):
        await error_function()
        
    profiler.stop_profiling()
    
    # 성능 통계에 기록되었는지 확인
    stats = profiler.get_performance_stats()
    assert 'execution_time' in stats
    assert stats['execution_time']['count'] == 1
    
@pytest.mark.asyncio
async def test_threshold_updates(profiler):
    """임계값 업데이트 테스트"""
    # 임계값 업데이트
    new_thresholds = {
        'execution_time': 0.3,
        'memory_usage': 30 * 1024 * 1024,
        'cpu_usage': 0.6,
        'latency': 0.03
    }
    profiler.thresholds.update(new_thresholds)
    
    # 함수 실행
    normal_func = profiler.profile_execution(normal_function)
    
    profiler.start_profiling()
    await normal_func()
    profiler.stop_profiling()
    
    # 새로운 임계값으로 경고가 발생하는지 확인
    report = profiler.generate_profile_report()
    assert len(report['warnings']) > 0  # 낮아진 임계값으로 인한 경고 발생
    
@pytest.mark.asyncio
async def test_long_running_profiling(profiler):
    """장시간 프로파일링 테스트"""
    # 장시간 실행 함수
    @profiler.profile_execution
    async def long_running_function():
        for _ in range(5):
            await asyncio.sleep(0.2)
            
    profiler.start_profiling()
    await long_running_function()
    profiler.stop_profiling()
    
    # 실행 시간 검증
    stats = profiler.get_performance_stats()
    assert stats['total_time'] >= 1.0  # 최소 1초 이상 실행
    
    # 보고서 생성 및 검증
    report = profiler.generate_profile_report()
    assert report['duration'] >= 1.0
    assert len(report['warnings']) > 0  # 긴 실행 시간으로 인한 경고 