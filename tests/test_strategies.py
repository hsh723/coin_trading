"""
전략 구현 모듈 테스트
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
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
def momentum_strategy(sample_data):
    """모멘텀 전략 인스턴스 생성"""
    return MomentumStrategy(
        data=sample_data,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )

@pytest.fixture
def mean_reversion_strategy(sample_data):
    """평균 회귀 전략 인스턴스 생성"""
    return MeanReversionStrategy(
        data=sample_data,
        bb_period=20,
        bb_std=2,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30
    )

@pytest.fixture
def breakout_strategy(sample_data):
    """브레이크아웃 전략 인스턴스 생성"""
    return BreakoutStrategy(
        data=sample_data,
        period=20,
        std_dev=2,
        volume_factor=1.5
    )

def test_base_strategy():
    """기본 전략 클래스 테스트"""
    strategy = BaseStrategy()
    
    with pytest.raises(NotImplementedError):
        strategy.generate_signals()
    
    with pytest.raises(NotImplementedError):
        strategy.calculate_position_size()

def test_momentum_strategy_signals(momentum_strategy):
    """모멘텀 전략 신호 생성 테스트"""
    signals = momentum_strategy.generate_signals()
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(momentum_strategy.data)
    assert signals.isin([-1, 0, 1]).all()
    assert not signals.isnull().any()

def test_mean_reversion_strategy_signals(mean_reversion_strategy):
    """평균 회귀 전략 신호 생성 테스트"""
    signals = mean_reversion_strategy.generate_signals()
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(mean_reversion_strategy.data)
    assert signals.isin([-1, 0, 1]).all()
    assert not signals.isnull().any()

def test_breakout_strategy_signals(breakout_strategy):
    """브레이크아웃 전략 신호 생성 테스트"""
    signals = breakout_strategy.generate_signals()
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(breakout_strategy.data)
    assert signals.isin([-1, 0, 1]).all()
    assert not signals.isnull().any()

def test_position_size_calculation(momentum_strategy):
    """포지션 크기 계산 테스트"""
    position_size = momentum_strategy.calculate_position_size(
        capital=10000,
        risk_per_trade=0.02,
        stop_loss=0.02
    )
    
    assert isinstance(position_size, float)
    assert position_size > 0
    assert position_size <= 10000

def test_risk_management(momentum_strategy):
    """리스크 관리 테스트"""
    stop_loss, take_profit = momentum_strategy.calculate_risk_levels(
        entry_price=100,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    assert isinstance(stop_loss, float)
    assert isinstance(take_profit, float)
    assert stop_loss < 100
    assert take_profit > 100

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
    
    with pytest.raises(ValueError):
        MomentumStrategy(data=empty_data)
    
    with pytest.raises(ValueError):
        MeanReversionStrategy(data=empty_data)
    
    with pytest.raises(ValueError):
        BreakoutStrategy(data=empty_data) 