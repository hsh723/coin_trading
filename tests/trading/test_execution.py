import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from src.execution.execution_manager import ExecutionManager

@pytest_asyncio.fixture
async def execution_manager():
    config = {
        'logging': {
            'level': 'INFO',
            'file': 'execution.log'
        },
        'risk_limit': 0.1,
        'max_slippage': 0.002,
        'max_risk_exposure': 0.1,
        'default_strategy': 'twap',
        'strategies': {
            'twap': {
                'class': 'TWAPStrategy',
                'params': {
                    'interval': 60,
                    'num_slices': 10
                }
            },
            'vwap': {
                'class': 'VWAPStrategy',
                'params': {
                    'window_size': 100,
                    'volume_profile': 'historical'
                }
            },
            'market': {
                'class': 'MarketStrategy',
                'params': {}
            },
            'limit': {
                'class': 'LimitStrategy',
                'params': {
                    'price_offset': 0.001
                }
            }
        },
        'market_monitor': {
            'update_interval': 1.0,
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
            'impact_threshold': 0.001,
            'reversion_threshold': 0.002,
            'timing_score_threshold': 0.7
        },
        'error_handler': {
            'max_retries': 3,
            'retry_delay': 1.0,
            'max_history_size': 1000
        },
        'notifier': {
            'enabled': False,
            'types': ['telegram'],
            'critical_threshold': 0.9,
            'warning_threshold': 0.7,
            'info_threshold': 0.5
        },
        'asset_cache': {
            'update_interval': 1.0,
            'cache_size': 1000,
            'cache_ttl': 60,
            'max_cache_size': 10000
        },
        'performance_metrics': {
            'collection_interval': 1.0,
            'history_size': 1000,
            'metrics_window': 100
        },
        'strategy_optimizer': {
            'update_interval': 1.0,
            'optimization_window': 100
        }
    }
    
    # 모의 객체 생성
    mock_market_monitor = AsyncMock()
    mock_execution_monitor = AsyncMock()
    mock_quality_monitor = AsyncMock()
    mock_error_handler = AsyncMock()
    mock_notifier = AsyncMock()
    mock_logger = AsyncMock()
    mock_asset_cache = AsyncMock()
    mock_performance_metrics = AsyncMock()
    mock_strategy_optimizer = AsyncMock()
    mock_strategy_factory = MagicMock()
    mock_position_manager = AsyncMock()
    
    # 모의 객체 초기화
    mock_market_monitor.initialize.return_value = None
    mock_execution_monitor.initialize.return_value = None
    mock_quality_monitor.initialize.return_value = None
    mock_error_handler.initialize.return_value = None
    mock_notifier.initialize.return_value = None
    mock_logger.initialize.return_value = None
    mock_asset_cache.initialize.return_value = None
    mock_performance_metrics.initialize.return_value = None
    mock_strategy_optimizer.initialize.return_value = None
    
    # 모의 객체 반환값 설정
    mock_market_monitor.get_market_data.return_value = {
        'success': True,
        'symbol': 'BTC/USDT',
        'price': 50000.0,
        'volume': 100.0,
        'timestamp': datetime.now()
    }
    
    mock_market_monitor.get_order_book.return_value = {
        'success': True,
        'symbol': 'BTC/USDT',
        'bids': [[49900.0, 1.0], [49800.0, 2.0]],
        'asks': [[50100.0, 1.0], [50200.0, 2.0]],
        'timestamp': datetime.now()
    }
    
    mock_position_manager.get_position.return_value = {
        'success': True,
        'symbol': 'BTC/USDT',
        'size': 0.1,
        'entry_price': 50000.0,
        'unrealized_pnl': 0.0
    }
    
    mock_position_manager.adjust_position.return_value = {
        'success': True,
        'symbol': 'BTC/USDT',
        'target_size': 0.01,
        'current_size': 0.1,
        'timestamp': datetime.now()
    }
    
    mock_performance_metrics.get_metrics.return_value = {
        'success': True,
        'metrics': {
            'success_rate': 0.95,
            'fill_rate': 0.98,
            'cost_efficiency': 0.99,
            'latency': 0.1
        },
        'timestamp': datetime.now()
    }
    
    mock_error_handler.get_error_stats.return_value = {
        'success': True,
        'stats': {
            'total_errors': 0,
            'error_rate': 0.0,
            'error_types': {}
        },
        'timestamp': datetime.now()
    }
    
    mock_asset_cache.get_subscribed_symbols.return_value = ['BTC/USDT']
    mock_asset_cache.is_price_valid.return_value = True
    mock_asset_cache.get_price.return_value = 50000.0
    mock_asset_cache.get_orderbook.return_value = {
        'bids': [[49900.0, 1.0], [49800.0, 2.0]],
        'asks': [[50100.0, 1.0], [50200.0, 2.0]]
    }
    
    # ExecutionManager 인스턴스 생성
    manager = ExecutionManager(config)
    
    # 모의 객체 주입
    manager.market_monitor = mock_market_monitor
    manager.execution_monitor = mock_execution_monitor
    manager.quality_monitor = mock_quality_monitor
    manager.error_handler = mock_error_handler
    manager.notifier = mock_notifier
    manager.logger = mock_logger
    manager.asset_cache = mock_asset_cache
    manager.performance_metrics_collector = mock_performance_metrics
    manager.strategy_optimizer = mock_strategy_optimizer
    manager.strategy_factory = mock_strategy_factory
    manager.position_manager = mock_position_manager
    
    yield manager

@pytest.mark.asyncio
async def test_execution_initialization(execution_manager):
    assert execution_manager is not None
    assert execution_manager.config is not None
    assert execution_manager.default_strategy == 'twap'

@pytest.mark.asyncio
async def test_order_placement(execution_manager):
    order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 0.001,
        'order_type': 'market'
    }
    result = await execution_manager.execute_order(order)
    assert result is not None
    assert 'success' in result
    assert 'order_id' in result

@pytest.mark.asyncio
async def test_order_cancellation(execution_manager):
    """주문 취소 테스트"""
    # 주문 생성
    order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 0.001,
        'order_type': 'limit',
        'price': 50000.0
    }
    
    # 주문 실행 시작
    execution_id = await execution_manager._generate_execution_id()
    execution_manager.active_executions[execution_id] = {
        'order': order,
        'status': 'pending',
        'start_time': datetime.now(),
        'end_time': None,
        'result': None,
        'strategy': 'twap'
    }
    
    # 주문 취소
    cancel_result = await execution_manager.cancel_order(execution_id)
    
    # 취소 결과 확인
    assert cancel_result['success'] == True
    assert cancel_result['order_id'] == execution_id
    assert cancel_result['status'] == 'cancelled'
    assert 'timestamp' in cancel_result
    
    # 실행 상태 확인
    execution = execution_manager.active_executions[execution_id]
    assert execution['status'] == 'cancelled'
    assert execution['end_time'] is not None

@pytest.mark.asyncio
async def test_position_management(execution_manager):
    # 포지션 조회
    position = await execution_manager.get_position('BTC/USDT')
    assert position is not None
    assert 'symbol' in position
    assert 'size' in position
    
    # 포지션 조정
    adjust_result = await execution_manager.adjust_position('BTC/USDT', 0.01)
    assert adjust_result is not None
    assert 'success' in adjust_result

@pytest.mark.asyncio
async def test_market_data(execution_manager):
    # 시장 데이터 조회
    market_data = await execution_manager.get_market_data('BTC/USDT')
    assert market_data is not None
    assert 'symbol' in market_data
    assert 'price' in market_data
    
    # 호가 정보 조회
    order_book = await execution_manager.get_order_book('BTC/USDT')
    assert order_book is not None
    assert 'symbol' in order_book
    assert 'bids' in order_book
    assert 'asks' in order_book

@pytest.mark.asyncio
async def test_risk_management(execution_manager):
    # 위험 검사
    order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 0.001,
        'order_type': 'market'
    }
    risk_check = await execution_manager.check_risk(order)
    assert risk_check is not None
    assert 'success' in risk_check
    assert 'risk_exposure' in risk_check
    
    # 위험 한도 설정
    limits = {
        'risk_limit': 0.2,
        'max_slippage': 0.003
    }
    set_limits_result = await execution_manager.set_risk_limits(limits)
    assert set_limits_result is not None
    assert 'success' in set_limits_result

@pytest.mark.asyncio
async def test_performance_metrics(execution_manager):
    """성능 메트릭 수집 테스트"""
    # 주문 생성
    order = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': 1.0,
        'order_type': 'market'
    }
    
    # 주문 실행 시작
    execution_id = await execution_manager._generate_execution_id()
    execution_manager.active_executions[execution_id] = {
        'order': order,
        'status': 'pending',
        'start_time': datetime.now(),
        'end_time': None,
        'result': None,
        'strategy': 'twap'
    }
    
    # 성능 메트릭 확인
    metrics_history = execution_manager.performance_metrics.get_metrics_history()
    initial_history_length = len(metrics_history)
    
    # 주문 취소
    await execution_manager.cancel_order(execution_id)
    
    # 취소 후 메트릭 확인
    metrics_history = execution_manager.performance_metrics.get_metrics_history()
    assert len(metrics_history) > initial_history_length
    
    # 메트릭 데이터 구조 확인
    latest_metrics = metrics_history[-1]['metrics']
    assert 'latency' in latest_metrics
    assert 'fill_rate' in latest_metrics
    assert 'slippage' in latest_metrics
    assert 'execution_cost' in latest_metrics
    assert 'success_rate' in latest_metrics
    
    # 성능 점수 확인
    score = execution_manager.performance_metrics.get_performance_score()
    assert 0.0 <= score <= 1.0
    
    # 실행 통계 확인
    stats = execution_manager.performance_metrics.execution_stats
    assert stats['total_executions'] > 0
    assert stats['successful_executions'] >= 0
    assert stats['failed_executions'] >= 0

@pytest.mark.asyncio
async def test_error_handling(execution_manager):
    # 오류 통계 조회
    error_stats = await execution_manager.get_error_stats()
    assert error_stats is not None
    assert 'success' in error_stats
    assert 'stats' in error_stats 