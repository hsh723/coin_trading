"""
성능 메트릭 수집기 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.execution.performance_metrics import PerformanceMetricsCollector

@pytest.fixture
def metrics_collector():
    """테스트용 메트릭 수집기 인스턴스 생성"""
    config = {
        'max_history_size': 100,
        'metrics_weights': {
            'latency': 0.2,
            'fill_rate': 0.3,
            'slippage': 0.2,
            'execution_cost': 0.2,
            'success_rate': 0.1
        },
        'performance_thresholds': {
            'latency': {'min': 0.0, 'max': 1.0},
            'fill_rate': {'min': 0.0, 'max': 1.0},
            'slippage': {'min': 0.0, 'max': 0.1},
            'execution_cost': {'min': 0.0, 'max': 0.1},
            'success_rate': {'min': 0.0, 'max': 1.0}
        }
    }
    return PerformanceMetricsCollector(config)

@pytest.mark.asyncio
async def test_initialization(metrics_collector):
    """초기화 테스트"""
    # 초기 상태 확인
    assert metrics_collector.is_collecting == False
    assert metrics_collector._collection_task is None
    assert len(metrics_collector.metrics_history) == 0
    assert metrics_collector.execution_stats['total_executions'] == 0
    
    # 초기화
    await metrics_collector.initialize()
    assert metrics_collector.is_collecting == True
    assert metrics_collector._collection_task is not None
    
    # 종료
    await metrics_collector.close()
    assert metrics_collector.is_collecting == False

@pytest.mark.asyncio
async def test_add_execution_metrics(metrics_collector):
    """실행 메트릭 추가 테스트"""
    # 성공적인 실행 메트릭 추가
    metrics = {
        'latency': 0.5,
        'fill_rate': 0.95,
        'slippage': 0.02,
        'execution_cost': 0.01,
        'success': True
    }
    metrics_collector.add_execution_metrics(metrics)
    
    assert metrics_collector.execution_stats['total_executions'] == 1
    assert metrics_collector.execution_stats['successful_executions'] == 1
    assert metrics_collector.execution_stats['failed_executions'] == 0
    assert metrics_collector.execution_stats['total_latency'] == 0.5
    assert metrics_collector.execution_stats['total_slippage'] == 0.02
    assert metrics_collector.execution_stats['total_cost'] == 0.01
    
    # 실패한 실행 메트릭 추가
    metrics = {
        'latency': 0.8,
        'fill_rate': 0.5,
        'slippage': 0.05,
        'execution_cost': 0.02,
        'success': False
    }
    metrics_collector.add_execution_metrics(metrics)
    
    assert metrics_collector.execution_stats['total_executions'] == 2
    assert metrics_collector.execution_stats['successful_executions'] == 1
    assert metrics_collector.execution_stats['failed_executions'] == 1

@pytest.mark.asyncio
async def test_metrics_history(metrics_collector):
    """메트릭 이력 테스트"""
    await metrics_collector.initialize()
    
    # 메트릭 추가
    metrics = {
        'latency': 0.5,
        'fill_rate': 0.95,
        'slippage': 0.02,
        'execution_cost': 0.01,
        'success': True
    }
    metrics_collector.add_execution_metrics(metrics)
    
    # 시간 범위 필터링 테스트
    start_time = datetime.now() - timedelta(minutes=5)
    end_time = datetime.now() + timedelta(minutes=5)
    
    history = metrics_collector.get_metrics_history(start_time, end_time)
    assert len(history) > 0
    
    # 이력 데이터 구조 확인
    for entry in history:
        assert 'timestamp' in entry
        assert 'metrics' in entry
        assert isinstance(entry['timestamp'], datetime)
        assert isinstance(entry['metrics'], dict)
        
    await metrics_collector.close()

@pytest.mark.asyncio
async def test_performance_score(metrics_collector):
    """성능 점수 계산 테스트"""
    # 메트릭 추가
    metrics = {
        'latency': 0.5,
        'fill_rate': 0.95,
        'slippage': 0.02,
        'execution_cost': 0.01,
        'success': True
    }
    metrics_collector.add_execution_metrics(metrics)
    
    # 성능 점수 계산
    score = metrics_collector.get_performance_score()
    assert 0.0 <= score <= 1.0
    
    # 메트릭 값이 임계값을 초과하는 경우
    metrics = {
        'latency': 2.0,
        'fill_rate': 0.5,
        'slippage': 0.2,
        'execution_cost': 0.2,
        'success': False
    }
    metrics_collector.add_execution_metrics(metrics)
    
    score = metrics_collector.get_performance_score()
    assert 0.0 <= score <= 1.0

@pytest.mark.asyncio
async def test_error_handling(metrics_collector):
    """오류 처리 테스트"""
    # 잘못된 메트릭 추가
    metrics = {
        'invalid_metric': 1.0
    }
    metrics_collector.add_execution_metrics(metrics)
    
    # 누락된 메트릭 추가
    metrics = {
        'latency': 0.5
    }
    metrics_collector.add_execution_metrics(metrics)
    
    # 성능 점수 계산
    score = metrics_collector.get_performance_score()
    assert 0.0 <= score <= 1.0 