"""
실행 모니터 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from src.execution.execution_monitor import ExecutionMonitor

@pytest.fixture
def config():
    return {
        'update_interval': 0.1,
        'window_size': 100,
        'latency_threshold': 1.0,
        'fill_rate_threshold': 0.95,
        'slippage_threshold': 0.001,
        'cost_threshold': 0.002
    }

@pytest.fixture
async def execution_monitor(config):
    monitor = ExecutionMonitor(config)
    await monitor.initialize()
    yield monitor
    await monitor.close()

@pytest.mark.asyncio
async def test_initialization(config):
    monitor = ExecutionMonitor(config)
    await monitor.initialize()
    
    assert monitor.update_interval == 0.1
    assert monitor.window_size == 100
    assert monitor.is_monitoring == True
    assert monitor.monitor_task is not None
    
    await monitor.close()

@pytest.mark.asyncio
async def test_metrics_update(execution_monitor):
    # 정상 실행
    execution_monitor.update_metrics(
        order_id='test_order_1',
        latency=0.5,
        fill_rate=1.0,
        slippage=0.0001,
        execution_cost=0.001,
        volume=1.0,
        timestamp=datetime.now(),
        is_successful=True
    )
    
    state = execution_monitor.get_execution_state()
    assert state['state'] == 'normal'
    assert state['metrics']['latency'] == 0.5
    assert state['metrics']['fill_rate'] == 1.0
    assert state['stats']['total_orders'] == 1
    assert state['stats']['successful_orders'] == 1

@pytest.mark.asyncio
async def test_execution_state_classification(execution_monitor):
    # 높은 지연 시간
    execution_monitor.update_metrics(
        order_id='test_order_2',
        latency=2.0,  # 임계값 초과
        fill_rate=1.0,
        slippage=0.0001,
        execution_cost=0.001,
        volume=1.0,
        timestamp=datetime.now(),
        is_successful=True
    )
    
    state = execution_monitor.get_execution_state()
    assert state['state'] == 'slow'

    # 낮은 체결률
    execution_monitor.update_metrics(
        order_id='test_order_3',
        latency=0.5,
        fill_rate=0.9,  # 임계값 미만
        slippage=0.0001,
        execution_cost=0.001,
        volume=1.0,
        timestamp=datetime.now(),
        is_successful=True
    )
    
    state = execution_monitor.get_execution_state()
    assert state['state'] == 'poor_fill'

    # 높은 슬리피지
    execution_monitor.update_metrics(
        order_id='test_order_4',
        latency=0.5,
        fill_rate=1.0,
        slippage=0.002,  # 임계값 초과
        execution_cost=0.001,
        volume=1.0,
        timestamp=datetime.now(),
        is_successful=True
    )
    
    state = execution_monitor.get_execution_state()
    assert state['state'] == 'high_slippage'

@pytest.mark.asyncio
async def test_execution_stats(execution_monitor):
    # 성공한 주문
    execution_monitor.update_metrics(
        order_id='test_order_5',
        latency=0.5,
        fill_rate=1.0,
        slippage=0.0001,
        execution_cost=0.001,
        volume=1.0,
        timestamp=datetime.now(),
        is_successful=True
    )
    
    # 실패한 주문
    execution_monitor.update_metrics(
        order_id='test_order_6',
        latency=0.5,
        fill_rate=0.0,
        slippage=0.0,
        execution_cost=0.0,
        volume=0.0,
        timestamp=datetime.now(),
        is_successful=False
    )
    
    state = execution_monitor.get_execution_state()
    assert state['stats']['total_orders'] == 2
    assert state['stats']['successful_orders'] == 1
    assert state['stats']['failed_orders'] == 1
    assert state['stats']['total_volume'] == 1.0

@pytest.mark.asyncio
async def test_average_metrics(execution_monitor):
    # 여러 실행 데이터 추가
    for i in range(5):
        execution_monitor.update_metrics(
            order_id=f'test_order_{i+7}',
            latency=0.5 + i * 0.1,
            fill_rate=1.0 - i * 0.01,
            slippage=0.0001 + i * 0.0001,
            execution_cost=0.001 + i * 0.0001,
            volume=1.0,
            timestamp=datetime.now() - timedelta(minutes=i),
            is_successful=True
        )
    
    # 전체 평균 메트릭 확인
    avg_metrics = execution_monitor.get_average_metrics()
    assert avg_metrics['avg_latency'] > 0
    assert 0 < avg_metrics['avg_fill_rate'] <= 1
    assert avg_metrics['avg_slippage'] > 0
    assert avg_metrics['avg_cost'] > 0
    
    # 시간 범위 지정 평균 메트릭 확인
    now = datetime.now()
    filtered_metrics = execution_monitor.get_average_metrics(
        start_time=now - timedelta(minutes=2),
        end_time=now
    )
    assert filtered_metrics['avg_latency'] > 0

@pytest.mark.asyncio
async def test_metrics_window(execution_monitor):
    # 윈도우 크기보다 많은 데이터 추가
    for i in range(execution_monitor.window_size + 10):
        execution_monitor.update_metrics(
            order_id=f'test_order_{i+12}',
            latency=0.5,
            fill_rate=1.0,
            slippage=0.0001,
            execution_cost=0.001,
            volume=1.0,
            timestamp=datetime.now() - timedelta(seconds=i),
            is_successful=True
        )
    
    # 메트릭 크기 확인
    for key in execution_monitor.metrics:
        assert len(execution_monitor.metrics[key]) == execution_monitor.window_size

@pytest.mark.asyncio
async def test_time_range_filtering(execution_monitor):
    # 시간별 데이터 추가
    times = []
    for i in range(5):
        timestamp = datetime.now() - timedelta(minutes=i)
        times.append(timestamp)
        execution_monitor.update_metrics(
            order_id=f'test_order_{i+22}',
            latency=0.5,
            fill_rate=1.0,
            slippage=0.0001,
            execution_cost=0.001,
            volume=1.0,
            timestamp=timestamp,
            is_successful=True
        )
    
    # 시간 범위 필터링
    indices = execution_monitor._get_time_range_indices(
        start_time=times[-2],
        end_time=times[0]
    )
    assert len(indices) == 3

@pytest.mark.asyncio
async def test_error_handling(execution_monitor):
    # 잘못된 메트릭 데이터
    execution_monitor.update_metrics(
        order_id='test_order_27',
        latency=-1.0,  # 잘못된 값
        fill_rate=2.0,  # 잘못된 값
        slippage=0.0001,
        execution_cost=0.001,
        volume=1.0,
        timestamp=datetime.now(),
        is_successful=True
    )
    
    # 기본값 확인
    state = execution_monitor.get_execution_state()
    assert state['state'] == 'unknown'
    assert state['metrics']['latency'] == -1.0  # 에러 처리는 상위 레벨에서 수행

@pytest.mark.asyncio
async def test_cleanup(execution_monitor):
    # 데이터 추가
    execution_monitor.update_metrics(
        order_id='test_order_28',
        latency=0.5,
        fill_rate=1.0,
        slippage=0.0001,
        execution_cost=0.001,
        volume=1.0,
        timestamp=datetime.now(),
        is_successful=True
    )
    
    # 모니터 종료
    await execution_monitor.close()
    
    # 상태 확인
    assert execution_monitor.is_monitoring == False 