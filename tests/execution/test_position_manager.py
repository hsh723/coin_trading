import pytest
import asyncio
from datetime import datetime
from src.execution.position_manager import PositionManager, Position

@pytest.fixture
def position_manager():
    """PositionManager 인스턴스 생성"""
    return PositionManager()

@pytest.fixture
def sample_execution_data():
    """샘플 실행 데이터"""
    return {
        'symbol': 'BTC/USDT',
        'size': 1.0,
        'price': 50000.0,
        'side': 'buy',
        'timestamp': datetime.now()
    }

@pytest.mark.asyncio
async def test_get_position_empty(position_manager):
    """빈 포지션 조회 테스트"""
    position = await position_manager.get_position('BTC/USDT')
    assert position['size'] == 0.0
    assert position['entry_price'] == 0.0
    assert position['unrealized_pnl'] == 0.0

@pytest.mark.asyncio
async def test_create_position(position_manager, sample_execution_data):
    """새로운 포지션 생성 테스트"""
    position = await position_manager.update_position(
        sample_execution_data['symbol'],
        sample_execution_data
    )
    assert position.symbol == sample_execution_data['symbol']
    assert position.size == sample_execution_data['size']
    assert position.entry_price == sample_execution_data['price']

@pytest.mark.asyncio
async def test_modify_position(position_manager, sample_execution_data):
    """포지션 수정 테스트"""
    # 초기 포지션 생성
    await position_manager.update_position(
        sample_execution_data['symbol'],
        sample_execution_data
    )
    
    # 포지션 수정
    modify_data = {
        'symbol': 'BTC/USDT',
        'size': 0.5,
        'price': 51000.0,
        'side': 'buy',
        'timestamp': datetime.now()
    }
    position = await position_manager.update_position(
        modify_data['symbol'],
        modify_data
    )
    
    assert position.size == 1.5  # 1.0 + 0.5
    assert position.entry_price == pytest.approx(50333.33, rel=1e-4)  # (50000 * 1.0 + 51000 * 0.5) / 1.5

@pytest.mark.asyncio
async def test_adjust_position(position_manager, sample_execution_data):
    """포지션 조정 테스트"""
    # 초기 포지션 생성
    await position_manager.update_position(
        sample_execution_data['symbol'],
        sample_execution_data
    )
    
    # 포지션 조정
    result = await position_manager.adjust_position('BTC/USDT', 2.0)
    assert result['size'] == 1.0  # 현재는 조정 로직이 구현되지 않았으므로 원래 크기 반환

@pytest.mark.asyncio
async def test_calculate_position_metrics(position_manager, sample_execution_data):
    """포지션 메트릭 계산 테스트"""
    # 초기 포지션 생성
    await position_manager.update_position(
        sample_execution_data['symbol'],
        sample_execution_data
    )
    
    # 현재 가격 업데이트
    update_data = {
        'symbol': 'BTC/USDT',
        'size': 0.0,
        'price': 51000.0,
        'side': 'buy',
        'timestamp': datetime.now()
    }
    position = await position_manager.update_position(
        update_data['symbol'],
        update_data
    )
    
    assert position.unrealized_pnl == pytest.approx(1000.0, rel=1e-4)  # (51000 - 50000) * 1.0 