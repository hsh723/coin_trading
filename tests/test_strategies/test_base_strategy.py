import pytest
from src.strategy.base_strategy import BaseStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.breakout import BreakoutStrategy

def test_base_strategy_initialization():
    # Example test for strategy initialization
    assert True  # Replace with actual test logic

def test_base_strategy_abstract_methods():
    """추상 메서드 구현 테스트"""
    # 추상 클래스이므로 직접 인스턴스화할 수 없음
    with pytest.raises(TypeError):
        strategy = BaseStrategy()
        strategy.analyze(None)
        strategy.execute(None)
        strategy.generate_signals(None)
        strategy.get_state()
        strategy.initialize(None)
        strategy.set_state({})
        strategy.update(None)

def test_momentum_strategy_implementation(sample_market_data):
    """모멘텀 전략 구현 테스트"""
    strategy = MomentumStrategy()
    strategy.initialize(sample_market_data)
    
    # 분석 메서드 테스트
    analysis = strategy.analyze(sample_market_data)
    assert 'rsi' in analysis
    assert 'macd' in analysis
    assert 'signal' in analysis
    
    # 신호 생성 테스트
    signals = strategy.generate_signals(sample_market_data)
    assert 'buy_signal' in signals
    assert 'sell_signal' in signals
    
    # 상태 관리 테스트
    state = strategy.get_state()
    assert isinstance(state, dict)
    strategy.set_state({'test': 'value'})
    assert strategy.get_state()['test'] == 'value'

def test_mean_reversion_strategy_implementation(sample_market_data):
    """평균회귀 전략 구현 테스트"""
    strategy = MeanReversionStrategy()
    strategy.initialize(sample_market_data)
    
    # 분석 메서드 테스트
    analysis = strategy.analyze(sample_market_data)
    assert 'sma' in analysis
    assert 'std' in analysis
    assert 'zscore' in analysis
    
    # 신호 생성 테스트
    signals = strategy.generate_signals(sample_market_data)
    assert 'buy_signal' in signals
    assert 'sell_signal' in signals

def test_breakout_strategy_implementation(sample_market_data):
    """브레이크아웃 전략 구현 테스트"""
    strategy = BreakoutStrategy()
    strategy.initialize(sample_market_data)
    
    # 분석 메서드 테스트
    analysis = strategy.analyze(sample_market_data)
    assert 'high' in analysis
    assert 'low' in analysis
    assert 'atr' in analysis
    
    # 신호 생성 테스트
    signals = strategy.generate_signals(sample_market_data)
    assert 'buy_signal' in signals
    assert 'sell_signal' in signals

def test_strategy_execution(sample_market_data):
    """전략 실행 테스트"""
    strategy = MomentumStrategy()
    strategy.initialize(sample_market_data)
    
    # 거래 실행 테스트
    trade = strategy.execute(sample_market_data)
    assert trade is not None
    assert 'action' in trade
    assert 'amount' in trade
    
    # 업데이트 테스트
    strategy.update(sample_market_data)
    assert strategy.get_state() is not None