"""
트레이딩 시뮬레이터 테스트
"""

import pytest
import asyncio
from decimal import Decimal
from src.trading.simulation import TradingSimulator

@pytest.fixture
def simulator():
    """테스트용 시뮬레이터 생성"""
    return TradingSimulator(initial_capital=10000.0)

@pytest.mark.asyncio
async def test_initialization(simulator):
    """초기화 테스트"""
    assert simulator.initial_capital == Decimal('10000.0')
    assert simulator.balance == Decimal('10000.0')
    assert len(simulator.positions) == 0
    assert len(simulator.trade_history) == 0

@pytest.mark.asyncio
async def test_buy_order(simulator):
    """매수 주문 테스트"""
    # BTC/USDT 매수 주문
    success = await simulator.execute_order(
        symbol='BTC/USDT',
        side='buy',
        amount=0.1,
        price=50000.0,
        leverage=1
    )
    
    assert success is True
    assert 'BTC/USDT' in simulator.positions
    assert simulator.positions['BTC/USDT']['amount'] == Decimal('0.1')
    assert simulator.positions['BTC/USDT']['entry_price'] == Decimal('50000.0')
    assert simulator.positions['BTC/USDT']['side'] == 'long'
    assert simulator.positions['BTC/USDT']['leverage'] == Decimal('1')
    
    # 수수료 계산 확인
    cost = Decimal('0.1') * Decimal('50000.0')
    fee = cost * simulator.fee_rate
    expected_balance = simulator.initial_capital - cost - fee
    assert simulator.balance == expected_balance

@pytest.mark.asyncio
async def test_sell_order(simulator):
    """매도 주문 테스트"""
    # 먼저 매수 주문
    await simulator.execute_order(
        symbol='BTC/USDT',
        side='buy',
        amount=0.1,
        price=50000.0,
        leverage=1
    )
    
    # BTC/USDT 매도 주문
    success = await simulator.execute_order(
        symbol='BTC/USDT',
        side='sell',
        amount=0.1,
        price=51000.0,
        leverage=1
    )
    
    assert success is True
    assert 'BTC/USDT' not in simulator.positions
    
    # 수익 계산 확인
    revenue = Decimal('0.1') * Decimal('51000.0')
    fee = revenue * simulator.fee_rate
    profit = (Decimal('51000.0') - Decimal('50000.0')) * Decimal('0.1')
    expected_balance = simulator.initial_capital - (Decimal('0.1') * Decimal('50000.0')) - (Decimal('0.1') * Decimal('50000.0') * simulator.fee_rate) + revenue - fee + profit
    assert simulator.balance == expected_balance

@pytest.mark.asyncio
async def test_insufficient_balance(simulator):
    """잔고 부족 테스트"""
    # 잔고보다 큰 주문
    success = await simulator.execute_order(
        symbol='BTC/USDT',
        side='buy',
        amount=1.0,
        price=20000.0,
        leverage=1
    )
    
    assert success is False
    assert simulator.balance == simulator.initial_capital
    assert len(simulator.positions) == 0

@pytest.mark.asyncio
async def test_position_value(simulator):
    """포지션 가치 계산 테스트"""
    # 매수 주문
    await simulator.execute_order(
        symbol='BTC/USDT',
        side='buy',
        amount=0.1,
        price=50000.0,
        leverage=1
    )
    
    # 포지션 가치 계산
    position_value = simulator.get_position_value('BTC/USDT', 51000.0)
    expected_value = (51000.0 - 50000.0) * 0.1
    assert position_value == expected_value

@pytest.mark.asyncio
async def test_account_summary(simulator):
    """계좌 요약 정보 테스트"""
    # 매수 주문
    await simulator.execute_order(
        symbol='BTC/USDT',
        side='buy',
        amount=0.1,
        price=50000.0,
        leverage=1
    )
    
    summary = simulator.get_account_summary()
    assert summary['initial_capital'] == 10000.0
    assert summary['open_positions'] == 1
    assert summary['total_trades'] == 1

@pytest.mark.asyncio
async def test_trade_history(simulator):
    """거래 내역 테스트"""
    # 매수 주문
    await simulator.execute_order(
        symbol='BTC/USDT',
        side='buy',
        amount=0.1,
        price=50000.0,
        leverage=1
    )
    
    assert len(simulator.trade_history) == 1
    trade = simulator.trade_history[0]
    assert trade['symbol'] == 'BTC/USDT'
    assert trade['side'] == 'buy'
    assert trade['amount'] == 0.1
    assert trade['price'] == 50000.0
    assert 'fee' in trade
    assert 'timestamp' in trade
    assert 'balance' in trade 