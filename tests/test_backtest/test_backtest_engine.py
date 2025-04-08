import pytest
import pandas as pd
from src.backtest.engine import BacktestEngine
from src.strategy.momentum import MomentumStrategy
from src.risk.risk_manager import RiskManager

def test_backtest_engine_initialization():
    """백테스팅 엔진 초기화 테스트"""
    engine = BacktestEngine()
    assert engine is not None
    assert engine.strategy is None
    assert engine.risk_manager is None

def test_backtest_engine_setup(sample_market_data):
    """백테스팅 엔진 설정 테스트"""
    engine = BacktestEngine()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    engine.setup(strategy, risk_manager)
    assert engine.strategy == strategy
    assert engine.risk_manager == risk_manager

def test_backtest_execution():
    # Example test for backtest execution
    assert True  # Replace with actual test logic

def test_backtest_with_empty_data():
    """빈 데이터로 백테스팅 테스트"""
    engine = BacktestEngine()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    engine.setup(strategy, risk_manager)
    empty_data = pd.DataFrame()
    
    with pytest.raises(ValueError):
        engine.run(empty_data, initial_balance=10000)

def test_backtest_with_invalid_balance():
    """잘못된 초기 잔고로 백테스팅 테스트"""
    engine = BacktestEngine()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    engine.setup(strategy, risk_manager)
    
    with pytest.raises(ValueError):
        engine.run(sample_market_data, initial_balance=-10000)
    
    with pytest.raises(ValueError):
        engine.run(sample_market_data, initial_balance=0)

def test_backtest_performance_metrics(sample_market_data):
    """백테스팅 성능 지표 테스트"""
    engine = BacktestEngine()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    engine.setup(strategy, risk_manager)
    results = engine.run(sample_market_data, initial_balance=10000)
    
    # 수익률 테스트
    assert results['total_return'] >= -1.0  # 최대 손실은 -100%
    assert results['sharpe_ratio'] is not None
    
    # 최대 낙폭 테스트
    assert 0 <= results['max_drawdown'] <= 1.0
    
    # 거래 횟수 테스트
    assert len(results['trades']) >= 0
    
    # 승률 테스트
    assert 0 <= results['win_rate'] <= 1.0

def test_backtest_with_different_timeframes(sample_market_data):
    """다른 시간대 데이터로 백테스팅 테스트"""
    engine = BacktestEngine()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    engine.setup(strategy, risk_manager)
    
    # 일봉 데이터
    daily_data = sample_market_data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    results_daily = engine.run(daily_data, initial_balance=10000)
    assert results_daily is not None
    
    # 4시간봉 데이터
    h4_data = sample_market_data.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    results_h4 = engine.run(h4_data, initial_balance=10000)
    assert results_h4 is not None

def test_backtest_with_commission(sample_market_data):
    """수수료를 고려한 백테스팅 테스트"""
    engine = BacktestEngine()
    strategy = MomentumStrategy()
    risk_manager = RiskManager()
    
    engine.setup(strategy, risk_manager)
    
    # 수수료 0.1%로 테스트
    results_with_commission = engine.run(
        sample_market_data,
        initial_balance=10000,
        commission=0.001
    )
    
    # 수수료 없이 테스트
    results_without_commission = engine.run(
        sample_market_data,
        initial_balance=10000,
        commission=0.0
    )
    
    # 수수료가 있는 경우 수익률이 더 낮아야 함
    assert results_with_commission['total_return'] <= results_without_commission['total_return']