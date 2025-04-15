import pytest
import pytest_asyncio
from src.execution.execution_manager import ExecutionManager
import os
import json
import time
import pandas as pd
import numpy as np

@pytest_asyncio.fixture
async def execution_manager():
    config = {
        'default_strategy': 'twap',
        'risk_limit': 0.1,
        'max_slippage': 0.002,
        'execution_timeout': 300,
        'retry_limit': 3,
        'concurrent_limit': 5,
        'max_risk_exposure': 0.1,
        'strategies': {
            'twap': {},
            'vwap': {},
            'is': {},
            'pov': {},
            'adaptive': {}
        },
        'error_handler': {
            'token': 'test_token',
            'chat_id': 'test_chat_id'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/execution.log'
        }
    }
    
    manager = ExecutionManager(config=config)
    await manager.initialize()
    yield manager
    await manager.close()

@pytest.mark.asyncio
async def test_execution_initialization(execution_manager):
    assert execution_manager is not None
    assert execution_manager.config is not None
    assert execution_manager.config['default_strategy'] == 'twap'

@pytest.mark.asyncio
async def test_execution_start_stop(execution_manager):
    assert execution_manager.is_running is True
    
    await execution_manager.close()
    assert execution_manager.is_running is False

@pytest.mark.asyncio
async def test_order_placement(execution_manager):
    # 주문 생성
    order = {
        'symbol': 'BTCUSDT',
        'order_type': 'limit',
        'side': 'buy',
        'quantity': 0.1,
        'price': 50000.0
    }
    
    result = await execution_manager.execute_order(order)
    
    assert result is not None
    assert 'order_id' in result
    assert 'symbol' in result
    assert 'order_type' in result
    assert 'side' in result
    assert 'quantity' in result
    assert 'price' in result
    assert 'status' in result

@pytest.mark.asyncio
async def test_order_cancellation(execution_manager):
    # 주문 생성 후 취소
    order = {
        'symbol': 'BTCUSDT',
        'order_type': 'limit',
        'side': 'buy',
        'quantity': 0.1,
        'price': 50000.0
    }
    
    result = await execution_manager.execute_order(order)
    order_id = result['order_id']
    
    cancel_result = await execution_manager.cancel_order(symbol='BTCUSDT', order_id=order_id)
    assert cancel_result is True

@pytest.mark.asyncio
async def test_position_management(execution_manager):
    # 포지션 정보 조회
    position = await execution_manager.get_position('BTCUSDT')
    
    assert position is not None
    assert 'symbol' in position
    assert 'size' in position
    assert 'entry_price' in position
    assert 'unrealized_pnl' in position
    assert 'leverage' in position
    
    # 포지션 조정
    result = await execution_manager.adjust_position(
        symbol='BTCUSDT',
        size=0.1,
        leverage=10
    )
    
    assert result is True

@pytest.mark.asyncio
async def test_market_data(execution_manager):
    # 시장 데이터 조회
    market_data = await execution_manager.get_market_data('BTCUSDT')
    
    assert market_data is not None
    assert 'symbol' in market_data
    assert 'price' in market_data
    assert 'volume' in market_data
    assert 'timestamp' in market_data
    
    # 호가창 조회
    order_book = await execution_manager.get_order_book('BTCUSDT')
    
    assert order_book is not None
    assert 'bids' in order_book
    assert 'asks' in order_book
    assert 'timestamp' in order_book

@pytest.mark.asyncio
async def test_risk_management(execution_manager):
    # 리스크 체크
    risk_check = await execution_manager.check_risk(
        symbol='BTCUSDT',
        order_type='limit',
        side='buy',
        quantity=0.1,
        price=50000.0
    )
    
    assert risk_check is not None
    assert 'is_allowed' in risk_check
    assert 'reason' in risk_check
    
    # 리스크 한도 설정
    result = await execution_manager.set_risk_limits(
        symbol='BTCUSDT',
        max_position_size=1.0,
        max_leverage=20,
        max_daily_trades=100
    )
    
    assert result is True

@pytest.mark.asyncio
async def test_performance_monitoring(execution_manager):
    # 성능 메트릭 조회
    metrics = await execution_manager.get_performance_metrics()
    
    assert metrics is not None
    assert 'order_count' in metrics
    assert 'trade_count' in metrics
    assert 'total_volume' in metrics
    assert 'total_pnl' in metrics
    assert 'latency' in metrics
    
    # 지연 시간 측정
    latency = await execution_manager.measure_latency()
    
    assert latency is not None
    assert isinstance(latency, float)
    assert latency >= 0

@pytest.mark.asyncio
async def test_error_handling(execution_manager):
    # 잘못된 주문 생성 시도
    with pytest.raises(Exception):
        await execution_manager.execute_order({
            'symbol': 'INVALID',
            'order_type': 'limit',
            'side': 'buy',
            'quantity': 0.1,
            'price': 50000.0
        })
    
    # 에러 통계 확인
    error_stats = await execution_manager.get_error_stats()
    assert error_stats is not None
    assert error_stats['error_count'] > 0 