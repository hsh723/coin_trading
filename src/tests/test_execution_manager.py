"""
실행 매니저 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from src.execution.execution_manager import ExecutionManager

@pytest.fixture
def config():
    return {
        'default_strategy': 'twap',
        'max_retries': 3,
        'retry_delay': 0.1,
        'fee_rate': 0.001,
        'account_value': 100000.0,
        'max_risk_exposure': 0.1,
        'market_monitor': {
            'update_interval': 0.1
        },
        'execution_monitor': {
            'update_interval': 0.1
        },
        'quality_monitor': {
            'update_interval': 0.1
        }
    }

@pytest.fixture
async def mocked_components():
    """모의 구성 요소 생성"""
    # 모의 객체 생성
    market_monitor = AsyncMock()
    market_monitor.get_market_state.return_value = {
        'state': 'normal',
        'metrics': {
            'volatility': 0.001,
            'spread': 0.0005,
            'liquidity': 50.0
        }
    }
    
    execution_monitor = AsyncMock()
    execution_monitor.get_execution_state.return_value = {
        'state': 'normal',
        'metrics': {
            'latency': 0.5,
            'fill_rate': 1.0,
            'slippage': 0.0001,
            'cost': 0.001
        },
        'stats': {
            'total_orders': 10,
            'successful_orders': 9,
            'failed_orders': 1,
            'total_volume': 10.0,
            'total_cost': 0.01
        }
    }
    
    quality_monitor = AsyncMock()
    quality_monitor.get_current_metrics.return_value = {
        'market_impact': 0.0001,
        'price_reversion': 0.0001,
        'timing_score': 0.95,
        'execution_cost': 0.001
    }
    quality_monitor.is_execution_quality_good.return_value = True
    
    error_handler = AsyncMock()
    notifier = AsyncMock()
    logger = AsyncMock()
    cache_manager = AsyncMock()
    
    return {
        'market_monitor': market_monitor,
        'execution_monitor': execution_monitor,
        'quality_monitor': quality_monitor,
        'error_handler': error_handler,
        'notifier': notifier,
        'logger': logger,
        'cache_manager': cache_manager
    }

@pytest.fixture
async def execution_manager(config, mocked_components):
    """모의 구성 요소를 사용하는 실행 매니저"""
    
    # ExecutionManager 패치
    with patch('src.execution.execution_manager.MarketStateMonitor', return_value=mocked_components['market_monitor']), \
         patch('src.execution.execution_manager.ExecutionMonitor', return_value=mocked_components['execution_monitor']), \
         patch('src.execution.execution_manager.ExecutionQualityMonitor', return_value=mocked_components['quality_monitor']), \
         patch('src.execution.execution_manager.ErrorHandler', return_value=mocked_components['error_handler']), \
         patch('src.execution.execution_manager.ExecutionNotifier', return_value=mocked_components['notifier']), \
         patch('src.execution.execution_manager.ExecutionLogger', return_value=mocked_components['logger']), \
         patch('src.execution.execution_manager.CacheManager', return_value=mocked_components['cache_manager']):
        
        manager = ExecutionManager(config)
        await manager.initialize()
        yield manager
        await manager.close()

@pytest.mark.asyncio
async def test_initialization(execution_manager, mocked_components):
    """초기화 테스트"""
    # 모든 구성 요소가 초기화되었는지 확인
    assert execution_manager.market_monitor == mocked_components['market_monitor']
    assert execution_manager.execution_monitor == mocked_components['execution_monitor']
    assert execution_manager.quality_monitor == mocked_components['quality_monitor']
    assert execution_manager.error_handler == mocked_components['error_handler']
    assert execution_manager.notifier == mocked_components['notifier']
    assert execution_manager.logger == mocked_components['logger']
    assert execution_manager.cache_manager == mocked_components['cache_manager']
    
    # 초기화 메서드가 호출되었는지 확인
    mocked_components['market_monitor'].initialize.assert_called_once()
    mocked_components['execution_monitor'].initialize.assert_called_once()
    mocked_components['quality_monitor'].initialize.assert_called_once()
    mocked_components['error_handler'].initialize.assert_called_once()
    mocked_components['notifier'].initialize.assert_called_once()
    mocked_components['logger'].initialize.assert_called_once()
    mocked_components['cache_manager'].initialize.assert_called_once()
    
    # 전략 등록 확인
    assert len(execution_manager.execution_strategies) == 6
    assert 'twap' in execution_manager.execution_strategies
    assert 'vwap' in execution_manager.execution_strategies
    assert 'market' in execution_manager.execution_strategies
    assert 'limit' in execution_manager.execution_strategies
    assert 'iceberg' in execution_manager.execution_strategies
    assert 'adaptive' in execution_manager.execution_strategies

@pytest.mark.asyncio
async def test_order_execution(execution_manager, mocked_components):
    """주문 실행 테스트"""
    # 모의 실행 결과 설정
    execution_strategy_mock = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order',
        'price': 50000.0,
        'executed_qty': 1.0,
        'market_price': 50000.0,
        'fill_rate': 1.0
    })
    
    # TWAP 전략 패치
    execution_manager._execute_twap = execution_strategy_mock
    
    # 주문 요청
    order_request = {
        'order_id': 'test_order',
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 1.0,
        'price': 50000.0
    }
    
    # 주문 실행
    result = await execution_manager.execute_order(order_request, 'twap')
    
    # 실행 전략이 호출되었는지 확인
    execution_strategy_mock.assert_called_once()
    
    # 결과 검증
    assert result['success'] == True
    assert result['order_id'] == 'test_order'
    assert 'execution_time' in result
    assert 'fill_rate' in result
    assert 'slippage' in result
    assert 'execution_cost' in result
    
    # 모니터링 메트릭 업데이트 확인
    mocked_components['execution_monitor'].update_metrics.assert_called_once()
    mocked_components['quality_monitor'].add_execution.assert_called_once()
    
    # 로깅 및 알림 확인
    mocked_components['logger'].log_execution.assert_called()
    mocked_components['notifier'].notify_execution.assert_called_once()

@pytest.mark.asyncio
async def test_order_execution_failure(execution_manager, mocked_components):
    """주문 실행 실패 테스트"""
    # 모의 실행 결과 설정 (실패)
    execution_strategy_mock = AsyncMock(return_value={
        'success': False,
        'order_id': 'test_order',
        'error': 'Execution failed'
    })
    
    # TWAP 전략 패치
    execution_manager._execute_twap = execution_strategy_mock
    
    # 주문 요청
    order_request = {
        'order_id': 'test_order',
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 1.0,
        'price': 50000.0
    }
    
    # 주문 실행
    result = await execution_manager.execute_order(order_request, 'twap')
    
    # 실행 전략이 호출되었는지 확인
    execution_strategy_mock.assert_called_once()
    
    # 결과 검증
    assert result['success'] == False
    assert result['order_id'] == 'test_order'
    assert 'error' in result
    
    # 오류 알림 확인
    mocked_components['notifier'].notify_error.assert_called_once()

@pytest.mark.asyncio
async def test_order_validation(execution_manager):
    """주문 검증 테스트"""
    # 유효한 주문
    valid_order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 1.0,
        'price': 50000.0
    }
    
    validated = await execution_manager._validate_order_request(valid_order)
    assert validated['symbol'] == 'BTC/USDT'
    assert validated['side'] == 'BUY'
    assert validated['quantity'] == 1.0
    assert validated['price'] == 50000.0
    assert 'risk_exposure' in validated
    
    # 필수 필드 누락
    invalid_order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY'
        # quantity 누락
    }
    
    with pytest.raises(ValueError):
        await execution_manager._validate_order_request(invalid_order)
    
    # 유효하지 않은 수량
    invalid_order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': -1.0
    }
    
    with pytest.raises(ValueError):
        await execution_manager._validate_order_request(invalid_order)

@pytest.mark.asyncio
async def test_risk_exposure_calculation(execution_manager):
    """위험 노출도 계산 테스트"""
    # 낮은 위험
    low_risk_order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 0.1,
        'price': 50000.0
    }
    
    risk = execution_manager._calculate_risk_exposure(low_risk_order)
    assert risk == 0.05  # (0.1 * 50000) / 100000
    
    # 높은 위험
    high_risk_order = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 0.5,
        'price': 50000.0
    }
    
    risk = execution_manager._calculate_risk_exposure(high_risk_order)
    assert risk == 0.25  # (0.5 * 50000) / 100000

@pytest.mark.asyncio
async def test_slippage_calculation(execution_manager):
    """슬리피지 계산 테스트"""
    # 매수 주문 (양의 슬리피지)
    buy_request = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 1.0,
        'price': 50000.0
    }
    
    buy_result = {
        'price': 50100.0  # 실행가가 요청가보다 높음
    }
    
    slippage = execution_manager._calculate_slippage(buy_request, buy_result)
    assert slippage == 0.002  # (50100 - 50000) / 50000
    
    # 매도 주문 (양의 슬리피지)
    sell_request = {
        'symbol': 'BTC/USDT',
        'side': 'SELL',
        'quantity': 1.0,
        'price': 50000.0
    }
    
    sell_result = {
        'price': 49900.0  # 실행가가 요청가보다 낮음
    }
    
    slippage = execution_manager._calculate_slippage(sell_request, sell_result)
    assert slippage == 0.002  # (50000 - 49900) / 50000

@pytest.mark.asyncio
async def test_execution_cost_calculation(execution_manager):
    """실행 비용 계산 테스트"""
    request = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'quantity': 1.0,
        'price': 50000.0
    }
    
    result = {
        'price': 50100.0,
        'executed_qty': 1.0
    }
    
    cost = execution_manager._calculate_execution_cost(request, result)
    
    # 수수료 비용: 50100 * 1.0 * 0.001 = 50.1
    # 슬리피지 비용: 50100 * 1.0 * 0.002 = 100.2
    # 총 비용: 150.3
    # 정규화: 150.3 / (50100 * 1.0) = 0.003
    assert cost == pytest.approx(0.003, 0.0001)

@pytest.mark.asyncio
async def test_retry_execution(execution_manager, mocked_components):
    """실행 재시도 테스트"""
    # 첫 번째 시도는 실패하고 두 번째 시도는 성공하는 모의 함수
    attempt_count = 0
    
    async def mock_execute_order(request, strategy):
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count == 1:
            return {
                'success': False,
                'order_id': 'test_order',
                'error': 'First attempt failed'
            }
        else:
            return {
                'success': True,
                'order_id': 'test_order',
                'price': 50000.0,
                'executed_qty': 1.0,
                'market_price': 50000.0,
                'fill_rate': 1.0
            }
    
    # execute_order 메서드 패치
    execution_manager.execute_order = AsyncMock(side_effect=mock_execute_order)
    
    # 활성 실행 추가
    execution_manager.active_executions['test_order'] = {
        'request': {
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'quantity': 1.0,
            'price': 50000.0
        },
        'strategy': 'twap',
        'start_time': datetime.now(),
        'status': 'failed'
    }
    
    # 재시도 실행
    result = await execution_manager.retry_execution('test_order', max_retries=2)
    
    # 두 번 시도되었는지 확인
    assert attempt_count == 2
    
    # 결과 검증
    assert result['success'] == True
    assert result['order_id'] == 'test_order'

@pytest.mark.asyncio
async def test_optimize_execution_strategy(execution_manager, mocked_components):
    """실행 전략 최적화 테스트"""
    # 다양한 시장 상태 테스트
    market_states = {
        'volatile': 'twap',
        'wide_spread': 'limit',
        'illiquid': 'iceberg',
        'turbulent': 'adaptive',
        'normal': 'vwap'
    }
    
    execution_state = {
        'state': 'normal',
        'metrics': {
            'latency': 0.5,
            'fill_rate': 1.0,
            'slippage': 0.0001,
            'cost': 0.001
        },
        'stats': {
            'total_orders': 10,
            'successful_orders': 9,
            'failed_orders': 1,
            'total_volume': 10.0,
            'total_cost': 0.01
        }
    }
    
    quality_state = {
        'market_impact': 0.0001,
        'price_reversion': 0.0001,
        'timing_score': 0.95,
        'execution_cost': 0.001
    }
    
    for state, expected_strategy in market_states.items():
        market_state = {
            'state': state,
            'metrics': {
                'volatility': 0.001,
                'spread': 0.0005,
                'liquidity': 50.0
            }
        }
        
        execution_manager._optimize_execution_strategy(
            market_state, execution_state, quality_state
        )
        
        assert execution_manager.default_strategy == expected_strategy

@pytest.mark.asyncio
async def test_detect_anomalies(execution_manager, mocked_components):
    """이상 탐지 테스트"""
    # 정상 상태
    normal_market_state = {
        'state': 'normal',
        'metrics': {
            'volatility': 0.001,
            'spread': 0.0005,
            'liquidity': 50.0
        }
    }
    
    normal_execution_state = {
        'state': 'normal',
        'metrics': {
            'latency': 0.5,
            'fill_rate': 1.0,
            'slippage': 0.0001,
            'cost': 0.001
        },
        'stats': {
            'total_orders': 10,
            'successful_orders': 9,
            'failed_orders': 1,
            'total_volume': 10.0,
            'total_cost': 0.01
        }
    }
    
    normal_quality_state = {
        'market_impact': 0.0001,
        'price_reversion': 0.0001,
        'timing_score': 0.95,
        'execution_cost': 0.001
    }
    
    # 이상 없는 경우
    execution_manager._detect_anomalies(
        normal_market_state, normal_execution_state, normal_quality_state
    )
    
    # 알림이 발송되지 않아야 함
    mocked_components['notifier'].notify_error.assert_not_called()
    mocked_components['notifier'].notify_performance.assert_not_called()
    
    # 이상 있는 경우
    anomaly_execution_state = {
        'state': 'slow',  # 이상 상태
        'metrics': {
            'latency': 2.0,
            'fill_rate': 1.0,
            'slippage': 0.0001,
            'cost': 0.001
        },
        'stats': {
            'total_orders': 10,
            'successful_orders': 9,
            'failed_orders': 1,
            'total_volume': 10.0,
            'total_cost': 0.01
        }
    }
    
    # 품질 모니터 패치
    mocked_components['quality_monitor'].is_execution_quality_good.return_value = False
    
    execution_manager._detect_anomalies(
        normal_market_state, anomaly_execution_state, normal_quality_state
    )
    
    # 알림이 발송되어야 함
    mocked_components['notifier'].notify_error.assert_called_once()
    mocked_components['notifier'].notify_performance.assert_called_once() 