"""
시장 상태 모니터 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from src.execution.market_state_monitor import MarketStateMonitor

@pytest.fixture
def config():
    return {
        'update_interval': 0.1,
        'window_size': 100,
        'volatility_threshold': 0.002,
        'spread_threshold': 0.001,
        'liquidity_threshold': 10.0
    }

@pytest.fixture
async def market_monitor(config):
    monitor = MarketStateMonitor(config)
    await monitor.initialize()
    yield monitor
    await monitor.close()

@pytest.mark.asyncio
async def test_initialization(config):
    monitor = MarketStateMonitor(config)
    await monitor.initialize()
    
    assert monitor.update_interval == 0.1
    assert monitor.window_size == 100
    assert monitor.is_monitoring == True
    assert monitor.monitor_task is not None
    
    await monitor.close()

@pytest.mark.asyncio
async def test_metrics_update(market_monitor):
    # 정상 시장 상태
    market_monitor.update_metrics(
        bid_price=50000.0,
        ask_price=50010.0,
        last_price=50005.0,
        volume=1.0,
        timestamp=datetime.now()
    )
    
    metrics = market_monitor.get_market_state()
    assert metrics['state'] == 'normal'
    assert metrics['metrics']['spread'] == 0.0002  # (50010 - 50000) / 50000
    assert metrics['metrics']['volume'] == 1.0

@pytest.mark.asyncio
async def test_volatility_calculation(market_monitor):
    # 여러 가격 데이터 추가
    prices = [50000.0, 50100.0, 50050.0, 50200.0, 50150.0]
    for price in prices:
        market_monitor.update_metrics(
            bid_price=price,
            ask_price=price + 10,
            last_price=price,
            volume=1.0,
            timestamp=datetime.now()
        )
    
    # 변동성 확인
    metrics = market_monitor.get_market_state()
    assert metrics['metrics']['volatility'] > 0

@pytest.mark.asyncio
async def test_market_state_classification(market_monitor):
    # 변동성이 높은 시장
    for i in range(5):
        price = 50000.0 + (i * 200)  # 큰 가격 변동
        market_monitor.update_metrics(
            bid_price=price,
            ask_price=price + 10,
            last_price=price,
            volume=1.0,
            timestamp=datetime.now()
        )
    
    state = market_monitor.get_market_state()
    assert state['state'] in ['volatile', 'turbulent']

    # 스프레드가 넓은 시장
    market_monitor.update_metrics(
        bid_price=50000.0,
        ask_price=50100.0,  # 0.2% 스프레드
        last_price=50050.0,
        volume=1.0,
        timestamp=datetime.now()
    )
    
    state = market_monitor.get_market_state()
    assert state['state'] in ['wide_spread', 'turbulent']

    # 유동성이 낮은 시장
    market_monitor.update_metrics(
        bid_price=50000.0,
        ask_price=50010.0,
        last_price=50005.0,
        volume=0.1,  # 낮은 거래량
        timestamp=datetime.now()
    )
    
    state = market_monitor.get_market_state()
    assert state['state'] == 'illiquid'

@pytest.mark.asyncio
async def test_average_metrics(market_monitor):
    # 여러 데이터 추가
    for i in range(5):
        market_monitor.update_metrics(
            bid_price=50000.0 + i,
            ask_price=50010.0 + i,
            last_price=50005.0 + i,
            volume=1.0 + i,
            timestamp=datetime.now() - timedelta(minutes=i)
        )
    
    # 전체 평균 메트릭 확인
    avg_metrics = market_monitor.get_average_metrics()
    assert avg_metrics['avg_spread'] > 0
    assert avg_metrics['avg_volatility'] >= 0
    assert avg_metrics['avg_liquidity'] > 0
    assert avg_metrics['avg_volume'] > 0
    
    # 시간 범위 지정 평균 메트릭 확인
    now = datetime.now()
    filtered_metrics = market_monitor.get_average_metrics(
        start_time=now - timedelta(minutes=2),
        end_time=now
    )
    assert filtered_metrics['avg_spread'] > 0

@pytest.mark.asyncio
async def test_metrics_window(market_monitor):
    # 윈도우 크기보다 많은 데이터 추가
    for i in range(market_monitor.window_size + 10):
        market_monitor.update_metrics(
            bid_price=50000.0,
            ask_price=50010.0,
            last_price=50005.0,
            volume=1.0,
            timestamp=datetime.now() - timedelta(seconds=i)
        )
    
    # 메트릭 크기 확인
    for key in market_monitor.metrics:
        assert len(market_monitor.metrics[key]) == market_monitor.window_size

@pytest.mark.asyncio
async def test_time_range_filtering(market_monitor):
    # 시간별 데이터 추가
    times = []
    for i in range(5):
        timestamp = datetime.now() - timedelta(minutes=i)
        times.append(timestamp)
        market_monitor.update_metrics(
            bid_price=50000.0,
            ask_price=50010.0,
            last_price=50005.0,
            volume=1.0,
            timestamp=timestamp
        )
    
    # 시간 범위 필터링
    indices = market_monitor._get_time_range_indices(
        start_time=times[-2],
        end_time=times[0]
    )
    assert len(indices) == 3

@pytest.mark.asyncio
async def test_error_handling(market_monitor):
    # 잘못된 가격 데이터
    market_monitor.update_metrics(
        bid_price=-1.0,  # 잘못된 값
        ask_price=50010.0,
        last_price=50005.0,
        volume=1.0,
        timestamp=datetime.now()
    )
    
    # 기본값 확인
    metrics = market_monitor.get_market_state()
    assert metrics['state'] == 'unknown'
    assert metrics['metrics']['spread'] == 0.0

@pytest.mark.asyncio
async def test_cleanup(market_monitor):
    # 데이터 추가
    market_monitor.update_metrics(
        bid_price=50000.0,
        ask_price=50010.0,
        last_price=50005.0,
        volume=1.0,
        timestamp=datetime.now()
    )
    
    # 모니터 종료
    await market_monitor.close()
    
    # 상태 확인
    assert market_monitor.is_monitoring == False 