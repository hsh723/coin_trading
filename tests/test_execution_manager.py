import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.execution.execution_manager import ExecutionManager
from src.exchange.binance_client import BinanceClient
from src.config.env_loader import EnvLoader

@pytest.fixture
def mock_env_loader():
    with patch('src.execution.execution_manager.EnvLoader') as mock:
        loader = Mock()
        loader.load = Mock()
        loader.get = Mock(side_effect=lambda key, default=None: {
            'EXCHANGE_API_KEY': 'test_key',
            'EXCHANGE_API_SECRET': 'test_secret',
            'EXCHANGE_NAME': 'binance',
            'TRADING_MODE': 'testnet',
            'NEWS_UPDATE_INTERVAL': 300,
            'NEWS_LANGUAGE': 'en',
            'NEWS_SENTIMENT_THRESHOLD': 0.5,
        }.get(key, default))
        mock.return_value = loader
        yield loader

@pytest.fixture
def mock_market_monitor():
    with patch('src.execution.execution_manager.MarketStateMonitor') as mock:
        monitor = AsyncMock()
        monitor.initialize = AsyncMock()
        mock.return_value = monitor
        yield monitor

@pytest.fixture
def mock_execution_monitor():
    with patch('src.execution.execution_manager.ExecutionMonitor') as mock:
        monitor = AsyncMock()
        monitor.initialize = AsyncMock()
        mock.return_value = monitor
        yield monitor

@pytest.fixture
def mock_quality_monitor():
    with patch('src.execution.execution_manager.ExecutionQualityMonitor') as mock:
        monitor = AsyncMock()
        monitor.initialize = AsyncMock()
        mock.return_value = monitor
        yield monitor

@pytest.fixture
def mock_error_handler():
    with patch('src.execution.execution_manager.ErrorHandler') as mock:
        handler = Mock()
        mock.return_value = handler
        yield handler

@pytest.fixture
def mock_notifier():
    with patch('src.execution.execution_manager.ExecutionNotifier') as mock:
        notifier = Mock()
        mock.return_value = notifier
        yield notifier

@pytest.fixture
def mock_asset_cache():
    with patch('src.execution.execution_manager.AssetCacheManager') as mock:
        cache = Mock()
        mock.return_value = cache
        yield cache

@pytest.fixture
def mock_performance_metrics():
    with patch('src.execution.execution_manager.PerformanceMetricsCollector') as mock:
        metrics = Mock()
        mock.return_value = metrics
        yield metrics

@pytest.fixture
def mock_strategy_optimizer():
    with patch('src.execution.execution_manager.ExecutionStrategyOptimizer') as mock:
        optimizer = Mock()
        mock.return_value = optimizer
        yield optimizer

@pytest.fixture
def mock_config():
    return {
        "trading": {
            "mode": "testnet",
            "max_position_size": 1000,
            "max_leverage": 3
        },
        "risk_management": {
            "max_drawdown": 0.1,
            "max_position_risk": 0.02
        }
    }

@pytest.fixture
def mock_binance_client():
    with patch('src.execution.execution_manager.BinanceClient') as mock:
        client = AsyncMock()
        client.initialize = AsyncMock()
        client.create_order = AsyncMock(return_value={
            "orderId": "test_order_id",
            "status": "FILLED",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 50000
        })
        client.cancel_order = AsyncMock(return_value={
            "orderId": "test_order_id",
            "status": "CANCELED"
        })
        client.get_order = AsyncMock(return_value={
            "orderId": "test_order_id",
            "status": "FILLED",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 50000
        })
        mock.return_value = client
        yield client

@pytest.mark.asyncio
async def test_initialize(mock_config, mock_binance_client, mock_env_loader,
                         mock_market_monitor, mock_execution_monitor, mock_quality_monitor,
                         mock_error_handler, mock_notifier, mock_asset_cache,
                         mock_performance_metrics, mock_strategy_optimizer):
    """초기화 테스트"""
    manager = ExecutionManager(mock_config)
    await manager.initialize()
    
    assert manager.exchange_client is not None
    assert manager.market_monitor is not None
    assert manager.execution_monitor is not None
    assert manager.quality_monitor is not None
    assert manager.error_handler is not None
    assert manager.notifier is not None
    assert manager.asset_cache is not None
    assert manager.performance_metrics is not None
    assert manager.strategy_optimizer is not None

@pytest.mark.asyncio
async def test_execute_order(mock_config, mock_binance_client, mock_env_loader,
                           mock_market_monitor, mock_execution_monitor, mock_quality_monitor,
                           mock_error_handler, mock_notifier, mock_asset_cache,
                           mock_performance_metrics, mock_strategy_optimizer):
    """주문 실행 테스트"""
    manager = ExecutionManager(mock_config)
    await manager.initialize()
    
    # 테스트 주문 실행
    order = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.1,
        "price": 50000
    }
    
    result = await manager.execute_order(order)
    
    assert result is not None
    assert "orderId" in result
    assert "status" in result
    assert result["status"] == "FILLED"

@pytest.mark.asyncio
async def test_cancel_order(mock_config, mock_binance_client, mock_env_loader,
                          mock_market_monitor, mock_execution_monitor, mock_quality_monitor,
                          mock_error_handler, mock_notifier, mock_asset_cache,
                          mock_performance_metrics, mock_strategy_optimizer):
    """주문 취소 테스트"""
    manager = ExecutionManager(mock_config)
    await manager.initialize()
    
    # 테스트 주문 취소
    result = await manager.cancel_order("BTCUSDT", "test_order_id")
    
    assert result is not None
    assert "orderId" in result
    assert "status" in result
    assert result["status"] == "CANCELED"

@pytest.mark.asyncio
async def test_get_order(mock_config, mock_binance_client, mock_env_loader,
                        mock_market_monitor, mock_execution_monitor, mock_quality_monitor,
                        mock_error_handler, mock_notifier, mock_asset_cache,
                        mock_performance_metrics, mock_strategy_optimizer):
    """주문 조회 테스트"""
    manager = ExecutionManager(mock_config)
    await manager.initialize()
    
    # 테스트 주문 조회
    result = await manager.get_order("BTCUSDT", "test_order_id")
    
    assert result is not None
    assert "orderId" in result
    assert "status" in result
    assert result["status"] == "FILLED" 