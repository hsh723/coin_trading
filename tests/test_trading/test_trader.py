import pytest
import pandas as pd
from datetime import datetime
from src.trading.trader import Trader
from src.strategy.momentum import MomentumStrategy
from src.risk.risk_manager import RiskManager

def test_trader_initialization():
    """트레이더 초기화 테스트"""
    trader = Trader()
    assert trader is not None
    assert trader.strategy is None
    assert trader.risk_manager is None
    assert trader.position == 0
    assert trader.balance == 0

def test_trader_setup():
    """트레이더 설정 테스트"""
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    assert trader.strategy == strategy
    assert trader.risk_manager == risk_manager

def test_execute_trade(sample_market_data, sample_trade):
    """거래 실행 테스트"""
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    
    # 매수 거래 실행
    trade_result = trader.execute_trade(sample_trade)
    assert trade_result is not None
    assert 'status' in trade_result
    assert 'order_id' in trade_result
    assert 'executed_price' in trade_result
    assert 'executed_amount' in trade_result
    
    # 포지션 업데이트 확인
    assert trader.position > 0
    assert trader.balance < 10000

def test_close_position(sample_market_data):
    """포지션 청산 테스트"""
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    trader.position = 0.1
    
    # 포지션 청산
    close_result = trader.close_position(sample_market_data.iloc[-1])
    assert close_result is not None
    assert 'status' in close_result
    assert 'order_id' in close_result
    assert 'executed_price' in close_result
    assert 'executed_amount' in close_result
    
    # 포지션 및 잔고 업데이트 확인
    assert trader.position == 0
    assert trader.balance > 10000

def test_update_state(sample_market_data):
    """상태 업데이트 테스트"""
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    trader.position = 0.1
    
    # 상태 업데이트
    trader.update_state(sample_market_data.iloc[-1])
    
    assert trader.last_price is not None
    assert trader.unrealized_pnl is not None
    assert trader.total_pnl is not None

def test_risk_management(sample_market_data):
    """리스크 관리 테스트"""
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    
    # 리스크 제한 초과 테스트
    large_trade = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': 100,  # 매우 큰 거래량
        'price': 100,
        'timestamp': int(datetime.now().timestamp() * 1000)
    }
    
    trade_result = trader.execute_trade(large_trade)
    assert trade_result['status'] == 'rejected'
    assert 'reason' in trade_result
    assert 'risk_limit' in trade_result['reason']

def test_trade_history(sample_market_data):
    """거래 기록 테스트"""
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    
    # 여러 거래 실행
    for _ in range(3):
        trade = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'amount': 0.1,
            'price': 100,
            'timestamp': int(datetime.now().timestamp() * 1000)
        }
        trader.execute_trade(trade)
    
    # 거래 기록 확인
    history = trader.get_trade_history()
    assert len(history) == 3
    assert all('order_id' in trade for trade in history)
    assert all('executed_price' in trade for trade in history)
    assert all('executed_amount' in trade for trade in history)

def test_performance_metrics(sample_market_data):
    """성능 지표 테스트"""
    trader = Trader()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    trader.setup(strategy, risk_manager)
    trader.balance = 10000
    
    # 거래 실행 및 성능 지표 확인
    trade = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': 0.1,
        'price': 100,
        'timestamp': int(datetime.now().timestamp() * 1000)
    }
    trader.execute_trade(trade)
    
    metrics = trader.get_performance_metrics()
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics 