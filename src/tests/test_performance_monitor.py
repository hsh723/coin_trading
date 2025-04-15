"""
성능 모니터링 모듈 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from src.execution.performance_monitor import PerformanceMonitor

@pytest.fixture
def config():
    """테스트 설정"""
    return {
        'performance': {
            'latency_threshold': 100.0,  # 100ms
            'success_rate_threshold': 0.95,  # 95%
            'fill_rate_threshold': 0.90,  # 90%
        },
        'logging': {
            'level': 'INFO',
            'file_path': 'logs/test_performance.log'
        }
    }

@pytest.fixture
def monitor(config):
    """성능 모니터 인스턴스"""
    return PerformanceMonitor(config)

@pytest.mark.asyncio
async def test_initialization(monitor):
    """초기화 테스트"""
    await monitor.initialize()
    
    assert monitor.latency_threshold == 100.0
    assert monitor.success_rate_threshold == 0.95
    assert monitor.fill_rate_threshold == 0.90
    assert len(monitor.latency_data) == 0
    assert len(monitor.execution_data) == 0
    assert len(monitor.fill_data) == 0
    
    await monitor.close()

@pytest.mark.asyncio
async def test_record_latency(monitor):
    """지연시간 기록 테스트"""
    await monitor.initialize()
    
    # 정상 지연시간
    monitor.record_latency(50.0)
    assert len(monitor.latency_data) == 1
    assert monitor.latency_data[0]['latency'] == 50.0
    
    # 임계값 초과 지연시간
    monitor.record_latency(150.0)
    assert len(monitor.latency_data) == 2
    assert monitor.latency_data[1]['latency'] == 150.0
    
    await monitor.close()

@pytest.mark.asyncio
async def test_record_execution(monitor):
    """실행 결과 기록 테스트"""
    await monitor.initialize()
    
    # 성공 케이스
    success_data = {
        'order_id': '123',
        'symbol': 'BTC/USDT',
        'success': True
    }
    monitor.record_execution(success_data)
    
    assert monitor.stats['total_executions'] == 1
    assert monitor.stats['successful_executions'] == 1
    
    # 실패 케이스
    failure_data = {
        'order_id': '456',
        'symbol': 'ETH/USDT',
        'success': False,
        'error': 'insufficient_funds'
    }
    monitor.record_execution(failure_data)
    
    assert monitor.stats['total_executions'] == 2
    assert monitor.stats['successful_executions'] == 1
    assert monitor.stats['failed_executions'] == 1
    
    await monitor.close()

@pytest.mark.asyncio
async def test_record_fill(monitor):
    """주문 체결 기록 테스트"""
    await monitor.initialize()
    
    # 완전 체결
    full_fill_data = {
        'order_id': '123',
        'symbol': 'BTC/USDT',
        'quantity': 1.0,
        'filled_quantity': 1.0,
        'filled': True
    }
    monitor.record_fill(full_fill_data)
    
    assert monitor.stats['total_orders'] == 1
    assert monitor.stats['filled_orders'] == 1
    assert monitor.stats['total_volume'] == 1.0
    assert monitor.stats['filled_volume'] == 1.0
    
    # 부분 체결
    partial_fill_data = {
        'order_id': '456',
        'symbol': 'ETH/USDT',
        'quantity': 2.0,
        'filled_quantity': 1.5,
        'filled': True
    }
    monitor.record_fill(partial_fill_data)
    
    assert monitor.stats['total_orders'] == 2
    assert monitor.stats['filled_orders'] == 2
    assert monitor.stats['total_volume'] == 3.0
    assert monitor.stats['filled_volume'] == 2.5
    
    await monitor.close()

@pytest.mark.asyncio
async def test_get_latency_stats(monitor):
    """지연시간 통계 조회 테스트"""
    await monitor.initialize()
    
    # 테스트 데이터 생성
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    for latency in latencies:
        monitor.record_latency(latency)
    
    # 전체 통계 조회
    stats = monitor.get_latency_stats()
    assert stats['min'] == 10.0
    assert stats['max'] == 50.0
    assert stats['mean'] == 30.0
    assert stats['median'] == 30.0
    assert stats['p95'] == 50.0
    assert stats['p99'] == 50.0
    
    # 시간 범위 지정 통계 조회
    window_stats = monitor.get_latency_stats(timedelta(minutes=1))
    assert window_stats['min'] == 10.0
    assert window_stats['max'] == 50.0
    
    await monitor.close()

@pytest.mark.asyncio
async def test_get_success_rate(monitor):
    """실행 성공률 조회 테스트"""
    await monitor.initialize()
    
    # 테스트 데이터 생성
    executions = [
        {'success': True},
        {'success': True},
        {'success': False},
        {'success': True},
        {'success': True}
    ]
    
    for execution in executions:
        monitor.record_execution(execution)
    
    # 성공률 확인
    success_rate = monitor.get_success_rate()
    assert success_rate == 0.8  # 4/5 = 0.8
    
    await monitor.close()

@pytest.mark.asyncio
async def test_get_fill_rate(monitor):
    """주문 체결률 조회 테스트"""
    await monitor.initialize()
    
    # 테스트 데이터 생성
    orders = [
        {'quantity': 1.0, 'filled_quantity': 1.0, 'filled': True},
        {'quantity': 2.0, 'filled_quantity': 1.5, 'filled': True},
        {'quantity': 1.0, 'filled_quantity': 0.8, 'filled': True}
    ]
    
    for order in orders:
        monitor.record_fill(order)
    
    # 체결률 확인
    fill_rate = monitor.get_fill_rate()
    assert abs(fill_rate - 0.825) < 0.001  # (1.0 + 1.5 + 0.8) / (1.0 + 2.0 + 1.0) = 0.825
    
    await monitor.close()

@pytest.mark.asyncio
async def test_get_performance_summary(monitor):
    """성능 요약 조회 테스트"""
    await monitor.initialize()
    
    # 테스트 데이터 생성
    monitor.record_latency(50.0)
    monitor.record_execution({'success': True})
    monitor.record_fill({
        'quantity': 1.0,
        'filled_quantity': 0.9,
        'filled': True
    })
    
    # 성능 요약 조회
    summary = monitor.get_performance_summary()
    
    assert 'latency' in summary
    assert 'success_rate' in summary
    assert 'fill_rate' in summary
    assert 'stats' in summary
    assert 'timestamp' in summary
    
    assert summary['success_rate'] == 1.0
    assert abs(summary['fill_rate'] - 0.9) < 0.001
    
    await monitor.close()

@pytest.mark.asyncio
async def test_reset_stats(monitor):
    """통계 초기화 테스트"""
    await monitor.initialize()
    
    # 테스트 데이터 생성
    monitor.record_latency(50.0)
    monitor.record_execution({'success': True})
    monitor.record_fill({
        'quantity': 1.0,
        'filled_quantity': 1.0,
        'filled': True
    })
    
    # 초기화 전 확인
    assert len(monitor.latency_data) > 0
    assert len(monitor.execution_data) > 0
    assert len(monitor.fill_data) > 0
    assert monitor.stats['total_executions'] > 0
    
    # 통계 초기화
    monitor.reset_stats()
    
    # 초기화 후 확인
    assert len(monitor.latency_data) == 0
    assert len(monitor.execution_data) == 0
    assert len(monitor.fill_data) == 0
    assert monitor.stats['total_executions'] == 0
    assert monitor.stats['total_volume'] == 0.0
    
    await monitor.close() 