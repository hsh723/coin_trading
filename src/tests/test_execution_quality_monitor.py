"""
실행 품질 모니터 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from src.execution.execution_quality_monitor import ExecutionQualityMonitor

@pytest.fixture
def config():
    return {
        'update_interval': 0.1,
        'window_size': 100,
        'impact_threshold': 0.001,
        'reversion_threshold': 0.002,
        'timing_score_threshold': 0.7
    }

@pytest.fixture
async def quality_monitor(config):
    monitor = ExecutionQualityMonitor(config)
    await monitor.initialize()
    yield monitor
    await monitor.close()

@pytest.mark.asyncio
async def test_initialization(config):
    monitor = ExecutionQualityMonitor(config)
    await monitor.initialize()
    
    assert monitor.update_interval == 0.1
    assert monitor.window_size == 100
    assert monitor.is_monitoring == True
    assert monitor.monitor_task is not None
    
    await monitor.close()

@pytest.mark.asyncio
async def test_metrics_collection(quality_monitor):
    # 실행 데이터 추가
    execution_data = {
        'execution_price': 50000.0,
        'market_price': 49990.0,
        'quantity': 1.0,
        'side': 'buy',
        'timestamp': datetime.now()
    }
    quality_monitor.add_execution(**execution_data)
    
    # 메트릭 확인
    metrics = quality_monitor.get_current_metrics()
    assert metrics['market_impact'] > 0
    assert metrics['price_reversion'] >= 0
    assert 0 <= metrics['timing_score'] <= 1
    assert metrics['execution_cost'] > 0

@pytest.mark.asyncio
async def test_market_impact_calculation(quality_monitor):
    # 매수 주문의 시장 충격
    buy_impact = quality_monitor._calculate_market_impact(
        execution_price=50010.0,
        market_price=50000.0,
        side='buy'
    )
    assert buy_impact == 0.0002  # (50010 - 50000) / 50000
    
    # 매도 주문의 시장 충격
    sell_impact = quality_monitor._calculate_market_impact(
        execution_price=49990.0,
        market_price=50000.0,
        side='sell'
    )
    assert sell_impact == 0.0002  # (50000 - 49990) / 50000

@pytest.mark.asyncio
async def test_price_reversion_calculation(quality_monitor):
    # 매수 주문의 가격 반전
    buy_reversion = quality_monitor._calculate_price_reversion(
        execution_price=50010.0,
        market_price=50000.0,
        side='buy'
    )
    assert buy_reversion == 0.0002  # (50000 - 50010) / 50010
    
    # 매도 주문의 가격 반전
    sell_reversion = quality_monitor._calculate_price_reversion(
        execution_price=49990.0,
        market_price=50000.0,
        side='sell'
    )
    assert sell_reversion == 0.0002  # (49990 - 50000) / 49990

@pytest.mark.asyncio
async def test_timing_score_calculation(quality_monitor):
    # 완벽한 타이밍
    perfect_score = quality_monitor._calculate_timing_score(
        execution_price=50000.0,
        market_price=50000.0,
        side='buy'
    )
    assert perfect_score == 1.0
    
    # 최악의 타이밍
    worst_score = quality_monitor._calculate_timing_score(
        execution_price=50100.0,
        market_price=50000.0,
        side='buy'
    )
    assert worst_score == 0.0

@pytest.mark.asyncio
async def test_execution_cost_calculation(quality_monitor):
    # 매수 주문의 실행 비용
    buy_cost = quality_monitor._calculate_execution_cost(
        execution_price=50010.0,
        market_price=50000.0,
        quantity=2.0,
        side='buy'
    )
    assert buy_cost == 20.0  # 2 * (50010 - 50000)
    
    # 매도 주문의 실행 비용
    sell_cost = quality_monitor._calculate_execution_cost(
        execution_price=49990.0,
        market_price=50000.0,
        quantity=2.0,
        side='sell'
    )
    assert sell_cost == 20.0  # 2 * (50000 - 49990)

@pytest.mark.asyncio
async def test_metrics_history(quality_monitor):
    # 여러 실행 데이터 추가
    for i in range(5):
        execution_data = {
            'execution_price': 50000.0 + i,
            'market_price': 50000.0,
            'quantity': 1.0,
            'side': 'buy',
            'timestamp': datetime.now() - timedelta(minutes=i)
        }
        quality_monitor.add_execution(**execution_data)
    
    # 전체 평균 메트릭 확인
    avg_metrics = quality_monitor.get_average_metrics()
    assert avg_metrics['avg_market_impact'] > 0
    assert avg_metrics['avg_price_reversion'] >= 0
    assert 0 <= avg_metrics['avg_timing_score'] <= 1
    assert avg_metrics['avg_execution_cost'] > 0
    
    # 시간 범위 지정 평균 메트릭 확인
    now = datetime.now()
    filtered_metrics = quality_monitor.get_average_metrics(
        start_time=now - timedelta(minutes=2),
        end_time=now
    )
    assert filtered_metrics['avg_market_impact'] > 0

@pytest.mark.asyncio
async def test_execution_quality_check(quality_monitor):
    # 좋은 실행 품질
    quality_monitor.add_execution(
        execution_price=50000.5,  # 0.001% 시장 충격
        market_price=50000.0,
        quantity=1.0,
        side='buy',
        timestamp=datetime.now()
    )
    assert quality_monitor.is_execution_quality_good() == True
    
    # 나쁜 실행 품질
    quality_monitor.add_execution(
        execution_price=50100.0,  # 0.2% 시장 충격
        market_price=50000.0,
        quantity=1.0,
        side='buy',
        timestamp=datetime.now()
    )
    assert quality_monitor.is_execution_quality_good() == False

@pytest.mark.asyncio
async def test_metrics_window(quality_monitor):
    # 윈도우 크기보다 많은 데이터 추가
    for i in range(quality_monitor.window_size + 10):
        execution_data = {
            'execution_price': 50000.0,
            'market_price': 50000.0,
            'quantity': 1.0,
            'side': 'buy',
            'timestamp': datetime.now() - timedelta(seconds=i)
        }
        quality_monitor.add_execution(**execution_data)
    
    # 메트릭 크기 확인
    for key in quality_monitor.metrics:
        assert len(quality_monitor.metrics[key]) == quality_monitor.window_size

@pytest.mark.asyncio
async def test_cleanup(quality_monitor):
    # 실행 데이터 추가
    execution_data = {
        'execution_price': 50000.0,
        'market_price': 50000.0,
        'quantity': 1.0,
        'side': 'buy',
        'timestamp': datetime.now()
    }
    quality_monitor.add_execution(**execution_data)
    
    # 모니터 종료
    await quality_monitor.close()
    
    # 상태 확인
    assert quality_monitor.is_monitoring == False
    assert len(quality_monitor.metrics['market_impact']) == 1 