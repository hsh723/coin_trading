"""
실행 매니저 테스트 모듈
"""

import pytest
import asyncio
from datetime import datetime
from src.execution.execution_manager import ExecutionManager
from src.exchange.binance_client import BinanceClient
from unittest.mock import AsyncMock, MagicMock, patch
import time

@pytest.fixture
def mock_config():
    return {
        'test_mode': True,  # 테스트 모드 활성화
        'max_position_size': 1000,
        'max_leverage': 5,
        'risk_limit_percent': 0.1,
        'market_monitor': {
            'update_interval': 1,
            'window_size': 100,
            'volatility_threshold': 0.002,
            'spread_threshold': 0.001,
            'liquidity_threshold': 10.0
        },
        'execution_monitor': {
            'update_interval': 1.0,
            'window_size': 100,
            'latency_threshold': 1.0,
            'fill_rate_threshold': 0.95,
            'slippage_threshold': 0.001,
            'cost_threshold': 0.002
        },
        'quality_monitor': {
            'update_interval': 1.0,
            'window_size': 100,
            'latency_threshold': 1.0,
            'fill_rate_threshold': 0.95,
            'slippage_threshold': 0.001,
            'cost_threshold': 0.002
        },
        'error_handler': {},
        'notifier': {},
        'logging': {},
        'asset_cache': {},
        'performance_metrics': {
            'max_history_size': 1000,
            'metrics_weights': {
                'latency': 0.2,
                'fill_rate': 0.3,
                'slippage': 0.2,
                'execution_cost': 0.2,
                'success_rate': 0.1
            }
        }
    }

@pytest.fixture
def mock_binance_client():
    """바이낸스 클라이언트 모의 객체"""
    mock_client = AsyncMock()
    
    # 초기화 관련 메서드
    mock_client.initialize.return_value = True
    mock_client.get_server_time.return_value = {'serverTime': int(time.time() * 1000)}
    mock_client.get_account.return_value = {'accountType': 'SPOT', 'balances': []}
    
    # 주문 관련 메서드
    mock_client.create_order.return_value = {
        'orderId': '12345',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'origQty': '0.001',
        'status': 'FILLED'
    }
    
    mock_client.cancel_order.return_value = {
        'orderId': '12345',
        'status': 'CANCELED'
    }
    
    mock_client.get_order.return_value = {
        'orderId': '12345',
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'origQty': '0.001',
        'status': 'FILLED',
        'transactTime': int(time.time() * 1000)
    }
    
    # 시장 데이터 관련 메서드
    mock_client.get_market_data.return_value = {
        'price': 50000.0,
        'bid': 49990.0,
        'ask': 50010.0,
        'volume': 100.0,
        'price_change_24h': -1.5
    }
    
    return mock_client

@pytest.fixture
def mock_market_monitor():
    """시장 모니터 모의 객체"""
    monitor = AsyncMock()
    monitor.get_market_data.return_value = {
        'price': 50000.0,
        'bid': 49990.0,
        'ask': 50010.0,
        'volume': 100.0,
        'price_change_24h': -1.5
    }
    monitor.get_order_book.return_value = {
        'bids': [[49990.0, 1.0]],
        'asks': [[50010.0, 1.0]],
        'timestamp': int(time.time() * 1000)
    }
    return monitor

@pytest.fixture
def mock_execution_monitor():
    """실행 모니터 모의 객체"""
    monitor = AsyncMock()
    monitor.get_metrics.return_value = {
        'latency': 0.1,
        'fill_rate': 0.95,
        'slippage': 0.001,
        'cost': 0.002
    }
    return monitor

@pytest.fixture
def mock_quality_monitor():
    """실행 품질 모니터 모의 객체"""
    monitor = AsyncMock()
    monitor.get_metrics.return_value = {
        'success_rate': 0.95,
        'error_rate': 0.05,
        'recovery_rate': 0.9
    }
    return monitor

@pytest.fixture
def mock_error_handler():
    """에러 핸들러 모의 객체"""
    handler = AsyncMock()
    handler.get_error_stats.return_value = {
        'total_errors': 0,
        'error_types': {},
        'recovery_rate': 0.95
    }
    return handler

@pytest.fixture
def mock_performance_metrics():
    """성능 메트릭 수집기 모의 객체"""
    metrics = AsyncMock()
    metrics.get_metrics.return_value = {
        'success_rate': 0.95,
        'fill_rate': 0.98,
        'cost_efficiency': 0.99,
        'latency': 0.1
    }
    return metrics

@pytest.fixture
async def execution_manager(mock_config, mock_binance_client, mock_market_monitor, mock_execution_monitor, mock_quality_monitor, mock_error_handler, mock_performance_metrics):
    """실행 관리자 fixture"""
    with patch('src.execution.execution_manager.BinanceClient', return_value=mock_binance_client), \
         patch('src.execution.execution_manager.MarketStateMonitor', return_value=mock_market_monitor), \
         patch('src.execution.execution_manager.ExecutionMonitor', return_value=mock_execution_monitor), \
         patch('src.execution.execution_manager.ExecutionQualityMonitor', return_value=mock_quality_monitor), \
         patch('src.execution.execution_manager.ErrorHandler', return_value=mock_error_handler), \
         patch('src.execution.execution_manager.PerformanceMetricsCollector', return_value=mock_performance_metrics):
        
        manager = ExecutionManager(mock_config)
        await manager.initialize()
        
        yield manager
        
        await manager.close()

@pytest.mark.asyncio
async def test_initialize(execution_manager):
    """초기화 테스트"""
    assert execution_manager.exchange_client is not None
    assert execution_manager.market_monitor is not None
    assert execution_manager.execution_monitor is not None
    assert execution_manager.quality_monitor is not None
    assert execution_manager.position_manager is not None

@pytest.mark.asyncio
async def test_execute_order(execution_manager):
    """주문 실행 테스트"""
    order = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'MARKET',
        'quantity': 0.1
    }
    
    result = await execution_manager.execute_order(order)
    assert result['status'] == 'FILLED'
    assert 'orderId' in result

@pytest.mark.asyncio
async def test_cancel_order(execution_manager):
    """주문 취소 테스트"""
    result = await execution_manager.cancel_order('BTCUSDT', '12345')
    assert result['status'] == 'CANCELED'
    assert result['orderId'] == '12345'

@pytest.mark.asyncio
async def test_get_position(execution_manager):
    """포지션 조회 테스트"""
    position = await execution_manager.get_position("BTC")
    assert position["size"] == 0.0
    assert position["entry_price"] == 0.0
    assert position["side"] is None

@pytest.mark.asyncio
async def test_adjust_position(execution_manager):
    """포지션 조정 테스트"""
    result = await execution_manager.adjust_position("BTC", 0.01)
    assert result["success"] is True
    assert result["symbol"] == "BTC"
    assert result["size"] == 0.01
    assert result["side"] == "LONG"

@pytest.mark.asyncio
async def test_get_market_data(execution_manager):
    """시장 데이터 조회 테스트"""
    data = await execution_manager.get_market_data('BTCUSDT')
    assert 'price' in data
    assert 'volume' in data
    assert 'price_change_24h' in data

@pytest.mark.asyncio
async def test_get_order_book(execution_manager):
    """호가창 조회 테스트"""
    order_book = await execution_manager.get_order_book("BTC")
    assert "bids" in order_book
    assert "asks" in order_book
    assert "timestamp" in order_book

@pytest.mark.asyncio
async def test_check_risk(execution_manager):
    """리스크 체크 테스트"""
    order = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 0.1,
        'price': 50000.0
    }
    risk = await execution_manager.check_risk(order)
    assert isinstance(risk, dict)
    assert 'exposure' in risk
    assert 'leverage' in risk
    assert 'risk_level' in risk

@pytest.mark.asyncio
async def test_get_performance_metrics(execution_manager):
    """성능 메트릭 조회 테스트"""
    metrics = await execution_manager.get_performance_metrics()
    assert "success_rate" in metrics
    assert "fill_rate" in metrics
    assert "cost_efficiency" in metrics
    assert "latency" in metrics

@pytest.mark.asyncio
async def test_get_error_stats(execution_manager):
    """에러 통계 조회 테스트"""
    stats = await execution_manager.get_error_stats()
    assert "total_errors" in stats
    assert "error_types" in stats
    assert "recovery_rate" in stats 