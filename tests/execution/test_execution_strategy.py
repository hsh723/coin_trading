"""
실행 전략 테스트 모듈
"""

import pytest
from datetime import datetime
from src.execution.strategies import (
    ExecutionStrategy,
    TWAPStrategy,
    VWAPStrategy,
    IcebergStrategy,
    AdaptiveStrategy
)

@pytest.fixture
def strategy_config():
    """전략 설정"""
    return {
        'time_window': 3600,  # 1시간
        'num_slices': 12,  # 12개 슬라이스
        'volume_window': 100,  # 100개 볼륨
        'display_size': 0.1,  # 표시 크기
        'refresh_interval': 60,  # 60초
        'initial_slice_size': 0.1,  # 초기 슬라이스 크기
        'max_slice_size': 0.5,  # 최대 슬라이스 크기
        'min_slice_size': 0.01,  # 최소 슬라이스 크기
        'adaptation_interval': 300  # 5분
    }

@pytest.fixture
def order_params():
    """주문 파라미터"""
    return {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 1.0,
        'price': 50000.0
    }

def test_twap_strategy_initialization(strategy_config):
    """TWAP 전략 초기화 테스트"""
    strategy = TWAPStrategy(strategy_config)
    assert strategy is not None
    assert strategy.time_window == 3600
    assert strategy.num_slices == 12

def test_vwap_strategy_initialization(strategy_config):
    """VWAP 전략 초기화 테스트"""
    strategy = VWAPStrategy(strategy_config)
    assert strategy is not None
    assert strategy.volume_window == 100
    assert strategy.num_slices == 10

def test_iceberg_strategy_initialization(strategy_config):
    """아이스버그 전략 초기화 테스트"""
    strategy = IcebergStrategy(strategy_config)
    assert strategy is not None
    assert strategy.display_size == 0.1
    assert strategy.refresh_interval == 60

def test_adaptive_strategy_initialization(strategy_config):
    """적응형 전략 초기화 테스트"""
    strategy = AdaptiveStrategy(strategy_config)
    assert strategy is not None
    assert strategy.initial_slice_size == 0.1
    assert strategy.max_slice_size == 0.5
    assert strategy.min_slice_size == 0.01

def test_twap_strategy_execution(strategy_config, order_params):
    """TWAP 전략 실행 테스트"""
    strategy = TWAPStrategy(strategy_config)
    result = strategy.execute(order_params)
    
    assert result is not None
    assert 'order_id' in result
    assert result['status'] == 'ACTIVE'
    assert result['total_quantity'] == order_params['quantity']
    assert result['remaining_quantity'] == order_params['quantity']

def test_vwap_strategy_execution(strategy_config, order_params):
    """VWAP 전략 실행 테스트"""
    strategy = VWAPStrategy(strategy_config)
    result = strategy.execute(order_params)
    
    assert result is not None
    assert 'order_id' in result
    assert result['status'] == 'ACTIVE'
    assert result['total_quantity'] == order_params['quantity']
    assert result['remaining_quantity'] == order_params['quantity']

def test_iceberg_strategy_execution(strategy_config, order_params):
    """아이스버그 전략 실행 테스트"""
    strategy = IcebergStrategy(strategy_config)
    result = strategy.execute(order_params)
    
    assert result is not None
    assert 'order_id' in result
    assert result['status'] == 'ACTIVE'
    assert result['total_quantity'] == order_params['quantity']
    assert result['remaining_quantity'] == order_params['quantity']

def test_adaptive_strategy_execution(strategy_config, order_params):
    """적응형 전략 실행 테스트"""
    strategy = AdaptiveStrategy(strategy_config)
    result = strategy.execute(order_params)
    
    assert result is not None
    assert 'order_id' in result
    assert result['status'] == 'ACTIVE'
    assert result['total_quantity'] == order_params['quantity']
    assert result['remaining_quantity'] == order_params['quantity']

def test_strategy_cancel(strategy_config, order_params):
    """전략 취소 테스트"""
    strategies = [
        TWAPStrategy(strategy_config),
        VWAPStrategy(strategy_config),
        IcebergStrategy(strategy_config),
        AdaptiveStrategy(strategy_config)
    ]
    
    for strategy in strategies:
        result = strategy.execute(order_params)
        order_id = result['order_id']
        assert strategy.cancel(order_id) is True

def test_strategy_status(strategy_config, order_params):
    """전략 상태 조회 테스트"""
    strategies = [
        TWAPStrategy(strategy_config),
        VWAPStrategy(strategy_config),
        IcebergStrategy(strategy_config),
        AdaptiveStrategy(strategy_config)
    ]
    
    for strategy in strategies:
        result = strategy.execute(order_params)
        order_id = result['order_id']
        status = strategy.get_status(order_id)
        
        assert status is not None
        assert 'order_id' in status
        assert status['status'] == 'ACTIVE'
        assert status['total_quantity'] == order_params['quantity'] 