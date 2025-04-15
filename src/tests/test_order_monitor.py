"""
주문 모니터 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.execution.order_monitor import OrderMonitor

@pytest.fixture
def config():
    return {
        'update_interval': 0.1,
        'history_size': 100,
        'latency_threshold': 1000,
        'fill_rate_threshold': 0.95,
        'slippage_threshold': 0.001,
        'retry_limit': 3
    }

@pytest.fixture
async def order_monitor(config):
    monitor = OrderMonitor(config)
    await monitor.initialize()
    yield monitor
    await monitor.close()

@pytest.mark.asyncio
async def test_initialization(config):
    monitor = OrderMonitor(config)
    await monitor.initialize()
    
    assert monitor.update_interval == 0.1
    assert monitor.history_size == 100
    assert monitor.is_monitoring == True
    assert monitor.monitor_task is not None
    
    await monitor.close()

@pytest.mark.asyncio
async def test_order_management(order_monitor):
    # 주문 추가
    order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'price': 50000.0
    }
    order_monitor.add_order('test_order', order)
    
    # 주문 조회
    active_order = order_monitor.get_order('test_order')
    assert active_order is not None
    assert active_order['status'] == 'new'
    assert active_order['retry_count'] == 0
    
    # 주문 업데이트
    updates = {'filled_quantity': 0.5}
    order_monitor.update_order('test_order', updates)
    
    updated_order = order_monitor.get_order('test_order')
    assert updated_order['filled_quantity'] == 0.5

@pytest.mark.asyncio
async def test_delayed_order(order_monitor):
    # 지연 주문 생성
    order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'price': 50000.0,
        'timestamp': datetime.now() - timedelta(seconds=2)
    }
    order_monitor.add_order('delayed_order', order)
    
    # 상태 업데이트 대기
    await asyncio.sleep(0.2)
    
    # 주문 상태 확인
    updated_order = order_monitor.get_order('delayed_order')
    assert updated_order['status'] == 'delayed'
    assert updated_order['retry_count'] == 1

@pytest.mark.asyncio
async def test_unfilled_order(order_monitor):
    # 미체결 주문 생성
    order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'price': 50000.0,
        'filled_quantity': 0.1
    }
    order_monitor.add_order('unfilled_order', order)
    
    # 상태 업데이트 대기
    await asyncio.sleep(0.2)
    
    # 주문 상태 확인
    updated_order = order_monitor.get_order('unfilled_order')
    assert updated_order['status'] == 'unfilled'

@pytest.mark.asyncio
async def test_failed_order(order_monitor):
    # 실패 주문 생성
    order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'price': 50000.0,
        'retry_count': 3
    }
    order_monitor.add_order('failed_order', order)
    
    # 상태 업데이트 대기
    await asyncio.sleep(0.2)
    
    # 주문 상태 확인
    failed_order = order_monitor.get_order('failed_order')
    assert failed_order is None
    
    # 이력 확인
    history = order_monitor.get_order_history()
    assert len(history) == 1
    assert history[0]['status'] == 'failed'

@pytest.mark.asyncio
async def test_order_history(order_monitor):
    # 여러 주문 생성
    for i in range(5):
        order = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 1.0,
            'price': 50000.0,
            'filled_quantity': 1.0,
            'status': 'filled',
            'timestamp': datetime.now() - timedelta(seconds=i)
        }
        order_monitor.add_order(f'order_{i}', order)
        order_monitor._archive_order(f'order_{i}')
    
    # 전체 이력 조회
    history = order_monitor.get_order_history()
    assert len(history) == 5
    
    # 시간 범위 지정 조회
    now = datetime.now()
    filtered_history = order_monitor.get_order_history(
        start_time=now - timedelta(seconds=2),
        end_time=now
    )
    assert len(filtered_history) == 3

@pytest.mark.asyncio
async def test_execution_statistics(order_monitor):
    # 성공 주문
    success_order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'target_price': 50000.0,
        'executed_price': 50010.0,
        'filled_quantity': 1.0,
        'status': 'filled',
        'timestamp': datetime.now() - timedelta(seconds=1),
        'filled_time': datetime.now()
    }
    order_monitor.add_order('success_order', success_order)
    order_monitor._archive_order('success_order')
    
    # 실패 주문
    failed_order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'target_price': 50000.0,
        'status': 'failed',
        'timestamp': datetime.now() - timedelta(seconds=1)
    }
    order_monitor.add_order('failed_order', failed_order)
    order_monitor._archive_order('failed_order')
    
    # 통계 확인
    stats = order_monitor.get_execution_statistics()
    assert stats['success_rate'] == 0.5  # 1/2
    assert stats['fill_rate'] == 0.5  # 1/2
    assert stats['avg_latency'] > 0
    assert stats['avg_slippage'] > 0

@pytest.mark.asyncio
async def test_cleanup(order_monitor):
    # 주문 추가
    order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'price': 50000.0
    }
    order_monitor.add_order('test_order', order)
    
    # 모니터 종료
    await order_monitor.close()
    
    # 상태 확인
    assert order_monitor.is_monitoring == False
    assert len(order_monitor.orders) == 1  # 주문은 유지 