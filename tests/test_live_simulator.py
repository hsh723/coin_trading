"""
실시간 시뮬레이터 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.trading.live_simulator import LiveSimulator

@pytest.fixture
async def simulator():
    """테스트용 시뮬레이터 생성"""
    simulator = LiveSimulator(exchange_name='binance')
    yield simulator
    await simulator.close()

@pytest.mark.asyncio
async def test_initialization(simulator):
    """초기화 테스트"""
    assert simulator.exchange_name == 'binance'
    assert simulator.market_data == {}
    assert simulator.is_running is False
    assert simulator.reconnect_attempts == 0

@pytest.mark.asyncio
async def test_websocket_connection(simulator):
    """WebSocket 연결 테스트"""
    with patch('websockets.connect') as mock_connect:
        mock_ws = Mock()
        mock_connect.return_value = mock_ws
        await simulator._connect_websocket()
        assert simulator.websocket == mock_ws
        assert simulator.reconnect_attempts == 0

@pytest.mark.asyncio
async def test_websocket_reconnection(simulator):
    """WebSocket 재연결 테스트"""
    with patch('websockets.connect') as mock_connect:
        # 첫 번째 연결 시도 실패
        mock_connect.side_effect = Exception("Connection failed")
        await simulator._connect_websocket()
        assert simulator.reconnect_attempts == 1

@pytest.mark.asyncio
async def test_market_data_processing(simulator):
    """시장 데이터 처리 테스트"""
    test_data = [
        {
            's': 'BTCUSDT',
            'c': '50000.0',
            'v': '100.0',
            'h': '51000.0',
            'l': '49000.0'
        }
    ]
    
    await simulator._process_market_data(test_data)
    assert 'BTCUSDT' in simulator.market_data
    assert simulator.market_data['BTCUSDT']['price'] == 50000.0
    assert simulator.market_data['BTCUSDT']['volume'] == 100.0

@pytest.mark.asyncio
async def test_run_simulation(simulator):
    """시뮬레이션 실행 테스트"""
    mock_strategy = Mock()
    mock_strategy.generate_signals.return_value = []
    
    simulator.is_running = True
    with patch('asyncio.create_task') as mock_create_task:
        mock_task = Mock()
        mock_create_task.return_value = mock_task
        
        try:
            await simulator.run_simulation(mock_strategy)
        except asyncio.CancelledError:
            pass
            
        assert mock_strategy.generate_signals.called
        assert mock_task.cancel.called

@pytest.mark.asyncio
async def test_get_market_data(simulator):
    """시장 데이터 조회 테스트"""
    simulator.market_data = {
        'BTCUSDT': {
            'price': 50000.0,
            'volume': 100.0
        }
    }
    
    data = simulator.get_market_data('BTCUSDT')
    assert data['price'] == 50000.0
    assert data['volume'] == 100.0
    
    data = simulator.get_market_data('ETHUSDT')
    assert data is None

@pytest.mark.asyncio
async def test_get_all_market_data(simulator):
    """전체 시장 데이터 조회 테스트"""
    simulator.market_data = {
        'BTCUSDT': {'price': 50000.0},
        'ETHUSDT': {'price': 3000.0}
    }
    
    data = simulator.get_all_market_data()
    assert len(data) == 2
    assert data['BTCUSDT']['price'] == 50000.0
    assert data['ETHUSDT']['price'] == 3000.0

@pytest.mark.asyncio
async def test_close(simulator):
    """리소스 정리 테스트"""
    mock_ws = Mock()
    mock_exchange = Mock()
    simulator.websocket = mock_ws
    simulator.exchange = mock_exchange
    
    await simulator.close()
    assert mock_ws.close.called
    assert mock_exchange.close.called 