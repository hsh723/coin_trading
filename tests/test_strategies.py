"""
전략 구현 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy
from src.strategy.breakout import BreakoutStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy
from src.analysis.indicators.technical import TechnicalIndicators

@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터 생성"""
    dates = pd.date_range(start='2021-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100),
        'rsi': np.random.uniform(0, 100, 100),
        'macd': np.random.normal(0, 1, 100),
        'macd_signal': np.random.normal(0, 1, 100),
        'bb_upper': np.random.normal(110, 5, 100),
        'bb_middle': np.random.normal(100, 5, 100),
        'bb_lower': np.random.normal(90, 5, 100)
    }, index=dates)
    return data

@pytest.fixture
def momentum_strategy():
    """모멘텀 전략 인스턴스 생성"""
    return MomentumStrategy(
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )

@pytest.fixture
def mean_reversion_strategy():
    """평균 회귀 전략 인스턴스 생성"""
    return MeanReversionStrategy(
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30
    )

@pytest.fixture
def breakout_strategy():
    """브레이크아웃 전략 인스턴스 생성"""
    return BreakoutStrategy(
        atr_period=14,
        atr_multiplier=2.0,
        min_volume=1000.0
    )

def test_base_strategy():
    """기본 전략 클래스 테스트"""
    with pytest.raises(TypeError):
        BaseStrategy()

def test_momentum_strategy_signals(momentum_strategy, sample_data):
    """모멘텀 전략 신호 생성 테스트"""
    momentum_strategy.initialize(sample_data)
    signals = momentum_strategy.generate_signals(sample_data)
    
    assert isinstance(signals, dict)
    assert 'buy' in signals
    assert 'sell' in signals
    assert 'analysis' in signals

def test_mean_reversion_strategy_signals(mean_reversion_strategy, sample_data):
    """평균 회귀 전략 신호 생성 테스트"""
    mean_reversion_strategy.initialize(sample_data)
    signals = mean_reversion_strategy.generate_signals(sample_data)
    
    assert isinstance(signals, dict)
    assert 'buy' in signals
    assert 'sell' in signals
    assert 'analysis' in signals

def test_breakout_strategy_signals(breakout_strategy, sample_data):
    """브레이크아웃 전략 신호 생성 테스트"""
    breakout_strategy.initialize(sample_data)
    signals = breakout_strategy.generate_signals(sample_data)
    
    assert isinstance(signals, dict)
    assert 'buy' in signals
    assert 'sell' in signals
    assert 'analysis' in signals

def test_position_size_calculation(momentum_strategy, sample_data):
    """포지션 크기 계산 테스트"""
    momentum_strategy.initialize(sample_data)
    result = momentum_strategy.execute(sample_data)
    
    assert isinstance(result, dict)
    assert 'action' in result
    assert result['action'] in ['buy', 'sell', 'hold']

def test_risk_management(momentum_strategy, sample_data):
    """리스크 관리 테스트"""
    momentum_strategy.initialize(sample_data)
    state = momentum_strategy.get_state()
    
    assert isinstance(state, dict)
    assert 'initialized' in state
    assert state['initialized'] is True

def test_strategy_parameters(momentum_strategy):
    """전략 파라미터 테스트"""
    assert momentum_strategy.rsi_period == 14
    assert momentum_strategy.rsi_overbought == 70
    assert momentum_strategy.rsi_oversold == 30
    assert momentum_strategy.macd_fast == 12
    assert momentum_strategy.macd_slow == 26
    assert momentum_strategy.macd_signal == 9

def test_empty_data():
    """빈 데이터 처리 테스트"""
    empty_data = pd.DataFrame()
    strategy = MomentumStrategy()
    
    with pytest.raises(ValueError):
        strategy.initialize(empty_data)
    
    with pytest.raises(ValueError):
        strategy.generate_signals(empty_data)
    
    with pytest.raises(ValueError):
        strategy.execute(empty_data) 