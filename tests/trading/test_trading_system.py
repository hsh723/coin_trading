"""
트레이딩 시스템 테스트 모듈
"""

import pytest
import asyncio
from src.trading.trading_system import TradingSystem
from unittest.mock import Mock, patch

@pytest.fixture
def trading_system():
    """TradingSystem 인스턴스 생성"""
    with patch('src.trading.trading_system.logger') as mock_logger:
        config = {
            'execution': {
                'max_order_size': 1.0,
                'min_order_size': 0.001,
                'max_slippage': 0.01,
                'max_cost': 0.002
            },
            'risk': {
                'position_limits': {
                    'max_size': 1.0,
                    'max_value': 100000.0,
                    'max_leverage': 5.0
                },
                'risk_limits': {
                    'max_order_size': 1.0,
                    'max_daily_loss': 0.1,
                    'max_drawdown': 0.2,
                    'liquidity': 1000.0,
                    'concentration': 0.3
                },
                'volatility_limits': {
                    'threshold': 0.02,
                    'window_size': 24,
                    'max_volatility': 0.05
                },
                'monitoring_interval': 1.0,
                'risk_thresholds': {
                    'position_size': 0.8,
                    'daily_loss': 0.05,
                    'drawdown': 0.1,
                    'volatility': 0.03
                }
            },
            'performance': {
                'window_size': 100,
                'update_interval': 1.0,
                'execution_weight': 0.4,
                'risk_weight': 0.3,
                'portfolio_weight': 0.3
            }
        }
        system = TradingSystem(config)
        yield system

@pytest.mark.asyncio
async def test_execute_trade(trading_system):
    """거래 실행 테스트"""
    # 시스템 초기화
    await trading_system.initialize()
    
    # 포지션 정보 설정
    position = {
        'symbol': 'BTC',
        'size': 0.01,
        'entry_price': 50000.0,
        'unrealized_pnl': 0.0,
        'side': 'buy'
    }
    await trading_system.position_manager.add_position(position)
    
    # 매수 주문
    order = {
        'symbol': 'BTC',
        'side': 'buy',
        'price': 50000.0,
        'size': 0.01,
        'order_type': 'market'
    }
    
    result = await trading_system.execute_trade(order)
    assert result['success'] is True
    
    # 활성 주문 확인
    active_orders = trading_system.get_active_orders()
    assert len(active_orders) == 1
    assert order['symbol'] in str(active_orders)
    
    # 매도 주문
    order = {
        'symbol': 'BTC',
        'side': 'sell',
        'price': 51000.0,
        'size': 0.01,
        'order_type': 'market'
    }
    
    result = await trading_system.execute_trade(order)
    
    assert result['success'] is True
    assert result['order_id'] is not None
    assert result['symbol'] == 'BTC'
    assert result['side'] == 'sell'
    
    # 시스템 종료
    await trading_system.close()

@pytest.mark.asyncio
async def test_cancel_order(trading_system):
    """주문 취소 테스트"""
    # 시스템 초기화
    await trading_system.initialize()
    
    # 포지션 정보 설정
    position = {
        'symbol': 'BTC',
        'size': 0.01,
        'entry_price': 50000.0,
        'unrealized_pnl': 0.0,
        'side': 'buy'
    }
    await trading_system.position_manager.add_position(position)
    
    # 매수 주문
    order = {
        'symbol': 'BTC',
        'side': 'buy',
        'price': 50000.0,
        'size': 0.01,
        'order_type': 'limit'
    }
    
    result = await trading_system.execute_trade(order)
    assert result['success'] is True
    order_id = result['order_id']
    
    # 주문 취소
    cancel_result = await trading_system.cancel_order(order_id)
    assert cancel_result['success'] is True
    
    # 주문 이력 확인
    order_history = trading_system.get_order_history()
    assert len(order_history) == 1
    assert order_history[0]['order']['symbol'] == 'BTC'
    
    # 시스템 종료
    await trading_system.close()

@pytest.mark.asyncio
async def test_get_risk_metrics(trading_system):
    """리스크 메트릭 조회 테스트"""
    # 시스템 초기화
    await trading_system.initialize()
    
    metrics = await trading_system.get_risk_metrics()
    
    assert isinstance(metrics, dict)
    assert 'position_risk' in metrics
    assert 'volatility_risk' in metrics
    assert 'liquidity_risk' in metrics
    assert 'concentration_risk' in metrics
    
    # 시스템 종료
    await trading_system.close()

@pytest.mark.asyncio
async def test_get_performance_metrics(trading_system):
    """성능 메트릭 조회 테스트"""
    # 시스템 초기화
    await trading_system.initialize()
    
    metrics = await trading_system.get_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert 'execution' in metrics
    assert 'risk' in metrics
    assert 'portfolio' in metrics
    assert 'performance' in metrics
    
    # 시스템 종료
    await trading_system.close()

def test_get_trade_history(trading_system):
    """거래 이력 조회 테스트"""
    history = trading_system.get_trade_history()
    assert isinstance(history, list)
    assert len(history) == 0  # 초기 상태에서는 빈 리스트

@pytest.mark.asyncio
async def test_risk_limit_exceeded(trading_system):
    """리스크 한도 초과 테스트"""
    # 시스템 초기화
    await trading_system.initialize()
    
    # 리스크 한도를 초과하는 거래
    order = {
        'symbol': 'BTC',
        'side': 'buy',
        'price': 50000.0,
        'size': 10.0,  # 매우 큰 수량
        'order_type': 'market'
    }
    
    position = {
        'symbol': 'BTC',
        'size': 0.0,
        'entry_price': 0.0,
        'side': 'long'
    }
    
    result = await trading_system.execute_trade(order, position)
    
    assert result['success'] is False
    assert result['error'] == 'risk_limit_exceeded' 