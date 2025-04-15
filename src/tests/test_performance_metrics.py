"""
성능 메트릭 수집기 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.execution.performance_metrics import PerformanceMetricsCollector

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'collection_interval': 0.1,  # 100ms
        'history_size': 10
    }

@pytest.fixture
async def metrics_collector(config):
    """메트릭 수집기 인스턴스"""
    collector = PerformanceMetricsCollector(config)
    await collector.initialize()
    yield collector
    await collector.close()

@pytest.mark.asyncio
async def test_initialization(config):
    """초기화 테스트"""
    collector = PerformanceMetricsCollector(config)
    await collector.initialize()
    
    # 설정 확인
    assert collector.collection_interval == 0.1
    assert collector.history_size == 10
    assert collector.is_collecting == True
    assert collector.collection_task is not None
    
    await collector.close()

@pytest.mark.asyncio
async def test_metrics_collection(metrics_collector):
    """메트릭 수집 테스트"""
    # 초기 메트릭 확인
    current_metrics = metrics_collector.get_current_metrics()
    assert current_metrics['latency'] == 0.0
    assert current_metrics['error_rate'] == 0.0
    assert current_metrics['fill_rate'] == 1.0
    assert current_metrics['slippage'] == 0.0
    assert current_metrics['throughput'] == 0.0
    
    # 메트릭 업데이트
    metrics_collector.update_execution_metrics(
        latency=100.0,
        success=True,
        filled=True,
        slippage=0.0005,
        volume=1.0
    )
    
    # 업데이트된 메트릭 확인
    current_metrics = metrics_collector.get_current_metrics()
    assert current_metrics['latency'] == 100.0
    assert current_metrics['error_rate'] == 0.0
    assert current_metrics['fill_rate'] == 1.0
    assert current_metrics['slippage'] == 0.0005
    
    # 실패 케이스 추가
    metrics_collector.update_execution_metrics(
        latency=150.0,
        success=False,
        filled=False,
        slippage=0.001,
        volume=0.5
    )
    
    # 업데이트된 메트릭 확인
    current_metrics = metrics_collector.get_current_metrics()
    assert current_metrics['latency'] == 150.0
    assert current_metrics['error_rate'] == 0.5  # 1/2
    assert current_metrics['fill_rate'] == 0.5  # 1/2
    assert current_metrics['slippage'] == 0.001

@pytest.mark.asyncio
async def test_metrics_history(metrics_collector):
    """메트릭 이력 테스트"""
    # 메트릭 업데이트
    for i in range(5):
        metrics_collector.update_execution_metrics(
            latency=100.0 + i * 10,
            success=True,
            filled=True,
            slippage=0.0001 * (i + 1),
            volume=1.0
        )
        await asyncio.sleep(0.2)  # 수집 간격 대기
    
    # 전체 이력 조회
    history = metrics_collector.get_metrics_history()
    assert len(history) > 0
    
    # 시간 범위 지정 조회
    now = datetime.now()
    start_time = now - timedelta(seconds=1)
    filtered_history = metrics_collector.get_metrics_history(
        start_time=start_time,
        end_time=now
    )
    assert len(filtered_history) <= len(history)

@pytest.mark.asyncio
async def test_statistics(metrics_collector):
    """통계 테스트"""
    # 초기 통계 확인
    stats = metrics_collector.get_statistics()
    assert stats['total_executions'] == 0
    assert stats['total_errors'] == 0
    assert stats['total_fills'] == 0
    assert stats['total_volume'] == 0.0
    
    # 실행 데이터 추가
    metrics_collector.update_execution_metrics(
        latency=100.0,
        success=True,
        filled=True,
        slippage=0.0005,
        volume=1.0
    )
    
    # 업데이트된 통계 확인
    stats = metrics_collector.get_statistics()
    assert stats['total_executions'] == 1
    assert stats['total_errors'] == 0
    assert stats['total_fills'] == 1
    assert stats['total_volume'] == 1.0
    
    # 실패 케이스 추가
    metrics_collector.update_execution_metrics(
        latency=150.0,
        success=False,
        filled=False,
        slippage=0.001,
        volume=0.5
    )
    
    # 업데이트된 통계 확인
    stats = metrics_collector.get_statistics()
    assert stats['total_executions'] == 2
    assert stats['total_errors'] == 1
    assert stats['total_fills'] == 1
    assert stats['total_volume'] == 1.5

@pytest.mark.asyncio
async def test_performance_score(metrics_collector):
    """성능 점수 테스트"""
    # 초기 점수 확인
    score = metrics_collector.calculate_performance_score()
    assert 0.0 <= score <= 1.0
    
    # 좋은 성능 메트릭
    metrics_collector.update_execution_metrics(
        latency=50.0,  # 50ms
        success=True,
        filled=True,
        slippage=0.0001,  # 0.01%
        volume=1.0
    )
    
    good_score = metrics_collector.calculate_performance_score()
    assert good_score > 0.8  # 80% 이상 기대
    
    # 나쁜 성능 메트릭
    metrics_collector.update_execution_metrics(
        latency=2000.0,  # 2초
        success=False,
        filled=False,
        slippage=0.005,  # 0.5%
        volume=0.1
    )
    
    bad_score = metrics_collector.calculate_performance_score()
    assert bad_score < 0.5  # 50% 미만 기대

@pytest.mark.asyncio
async def test_cleanup(metrics_collector):
    """정리 테스트"""
    # 메트릭 업데이트
    metrics_collector.update_execution_metrics(
        latency=100.0,
        success=True,
        filled=True,
        slippage=0.0005,
        volume=1.0
    )
    
    # 수집기 종료
    await metrics_collector.close()
    
    # 상태 확인
    assert metrics_collector.is_collecting == False
    assert len(metrics_collector.metrics_history) > 0  # 이력은 유지 