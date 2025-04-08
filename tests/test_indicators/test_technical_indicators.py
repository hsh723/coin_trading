import pytest
import pandas as pd
import numpy as np
from src.analysis.indicators.technical import TechnicalIndicators

def test_sma_calculation():
    """단순 이동평균 계산 테스트"""
    data = pd.Series([1, 2, 3, 4, 5])
    sma = TechnicalIndicators.calculate_sma(data, period=3)
    
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(sma, expected)

def test_ema_calculation():
    """지수 이동평균 계산 테스트"""
    data = pd.Series([1, 2, 3, 4, 5])
    ema = TechnicalIndicators.calculate_ema(data, period=3)
    
    # EMA는 초기값이 다르므로 첫 번째 값만 확인
    assert not np.isnan(ema.iloc[0])
    assert ema.iloc[-1] > data.iloc[-2]

def test_rsi_calculation():
    # Example test for RSI calculation
    assert True  # Replace with actual test logic

def test_macd_calculation():
    """MACD 계산 테스트"""
    data = pd.Series([100, 102, 104, 103, 105, 107, 106])
    macd, signal = TechnicalIndicators.calculate_macd(
        data,
        fast_period=3,
        slow_period=5,
        signal_period=2
    )
    
    assert not np.isnan(macd.iloc[-1])
    assert not np.isnan(signal.iloc[-1])
    assert len(macd) == len(data)
    assert len(signal) == len(data)

def test_bollinger_bands_calculation():
    """볼린저 밴드 계산 테스트"""
    data = pd.Series([100, 102, 104, 103, 105, 107, 106])
    upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
        data,
        period=3,
        std_dev=2
    )
    
    assert not np.isnan(upper.iloc[-1])
    assert not np.isnan(middle.iloc[-1])
    assert not np.isnan(lower.iloc[-1])
    assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]

def test_atr_calculation():
    """ATR 계산 테스트"""
    high = pd.Series([105, 107, 106, 108, 110])
    low = pd.Series([100, 102, 103, 104, 105])
    close = pd.Series([102, 104, 105, 106, 108])
    
    atr = TechnicalIndicators.calculate_atr(high, low, close, period=3)
    
    assert not np.isnan(atr.iloc[-1])
    assert atr.iloc[-1] > 0

def test_indicators_with_empty_data():
    """빈 데이터로 지표 계산 테스트"""
    data = pd.Series([])
    
    with pytest.raises(ValueError):
        TechnicalIndicators.calculate_sma(data, period=3)
    
    with pytest.raises(ValueError):
        TechnicalIndicators.calculate_ema(data, period=3)
    
    with pytest.raises(ValueError):
        TechnicalIndicators.calculate_rsi(data, period=3)

def test_indicators_with_invalid_period():
    """잘못된 기간으로 지표 계산 테스트"""
    data = pd.Series([1, 2, 3])
    
    with pytest.raises(ValueError):
        TechnicalIndicators.calculate_sma(data, period=0)
    
    with pytest.raises(ValueError):
        TechnicalIndicators.calculate_ema(data, period=-1)
    
    with pytest.raises(ValueError):
        TechnicalIndicators.calculate_rsi(data, period=len(data) + 1)