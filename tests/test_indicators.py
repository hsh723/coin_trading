import pytest
import pandas as pd
import numpy as np
from src.data.processor import DataProcessor
from src.indicators.technical import TechnicalIndicators

@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터 생성"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = pd.DataFrame({
        'open': np.random.normal(50000, 1000, 100),
        'high': np.random.normal(51000, 1000, 100),
        'low': np.random.normal(49000, 1000, 100),
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(100, 20, 100)
    }, index=dates)
    
    # OHLCV 데이터 일관성 보장
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 100, 100))
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 100, 100))
    data['volume'] = abs(data['volume'])
    
    return data

@pytest.fixture
def processor():
    """테스트용 데이터 프로세서 생성"""
    return DataProcessor(
        symbol="BTC/USDT",
        timeframe="1h"
    )

@pytest.fixture
def indicators(sample_data):
    """지표 계산기 인스턴스 생성"""
    return TechnicalIndicators(sample_data)

def test_processor_initialization(processor):
    """데이터 프로세서 초기화 테스트"""
    assert processor.symbol == "BTC/USDT"
    assert processor.timeframe == "1h"

def test_process_data(processor, sample_data):
    """데이터 전처리 테스트"""
    processed_data = processor.process_data(
        data=sample_data,
        handle_missing=True,
        handle_outliers=True,
        normalize=True
    )
    
    # 데이터 구조 검증
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == len(sample_data)
    assert all(col in processed_data.columns for col in sample_data.columns)
    
    # 결측치 처리 검증
    assert not processed_data.isnull().any().any()
    
    # 이상치 처리 검증
    z_scores = np.abs((processed_data - processed_data.mean()) / processed_data.std())
    assert not (z_scores > 3).any().any()
    
    # 정규화 검증
    assert processed_data['close'].mean() < 1
    assert processed_data['close'].std() < 1

def test_calculate_indicators(processor, sample_data):
    """기술 지표 계산 테스트"""
    # 기본 파라미터 설정
    params = {
        'bb_window': 20,
        'bb_std': 2.0,
        'rsi_window': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    # 지표 계산
    indicators = processor.calculate_indicators(sample_data, params)
    
    # 볼린저 밴드 검증
    assert 'bb_upper' in indicators.columns
    assert 'bb_lower' in indicators.columns
    assert 'bb_middle' in indicators.columns
    assert (indicators['bb_upper'] >= indicators['bb_middle']).all()
    assert (indicators['bb_lower'] <= indicators['bb_middle']).all()
    
    # RSI 검증
    assert 'rsi' in indicators.columns
    assert (indicators['rsi'] >= 0).all()
    assert (indicators['rsi'] <= 100).all()
    
    # MACD 검증
    assert 'macd' in indicators.columns
    assert 'macd_signal' in indicators.columns
    assert 'macd_hist' in indicators.columns

def test_handle_missing_data(processor, sample_data):
    """결측치 처리 테스트"""
    # 결측치 생성
    sample_data.loc[sample_data.index[10:20], 'close'] = np.nan
    
    # 결측치 처리
    processed_data = processor.process_data(
        data=sample_data,
        handle_missing=True,
        handle_outliers=False,
        normalize=False
    )
    
    # 결측치 처리 검증
    assert not processed_data.isnull().any().any()

def test_handle_outliers(processor, sample_data):
    """이상치 처리 테스트"""
    # 이상치 생성
    sample_data.loc[sample_data.index[10], 'close'] = 100000
    
    # 이상치 처리
    processed_data = processor.process_data(
        data=sample_data,
        handle_missing=False,
        handle_outliers=True,
        normalize=False
    )
    
    # 이상치 처리 검증
    z_scores = np.abs((processed_data - processed_data.mean()) / processed_data.std())
    assert not (z_scores > 3).any().any()

def test_normalize_data(processor, sample_data):
    """데이터 정규화 테스트"""
    # 데이터 정규화
    normalized_data = processor.process_data(
        data=sample_data,
        handle_missing=False,
        handle_outliers=False,
        normalize=True
    )
    
    # 정규화 검증
    assert normalized_data['close'].mean() < 1
    assert normalized_data['close'].std() < 1
    assert normalized_data['volume'].mean() < 1
    assert normalized_data['volume'].std() < 1

def test_rsi(indicators):
    """RSI 계산 테스트"""
    rsi = indicators.calculate_rsi(period=14)
    
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(indicators.data)
    assert rsi.min() >= 0
    assert rsi.max() <= 100
    assert not rsi.isnull().any()

def test_bollinger_bands(indicators):
    """볼린저 밴드 계산 테스트"""
    upper, middle, lower = indicators.calculate_bollinger_bands(period=20, std_dev=2)
    
    assert isinstance(upper, pd.Series)
    assert isinstance(middle, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(upper) == len(indicators.data)
    assert all(upper >= middle)
    assert all(middle >= lower)
    assert not upper.isnull().any()
    assert not middle.isnull().any()
    assert not lower.isnull().any()

def test_moving_averages(indicators):
    """이동평균 계산 테스트"""
    ma7 = indicators.calculate_ma(period=7)
    ma25 = indicators.calculate_ma(period=25)
    
    assert isinstance(ma7, pd.Series)
    assert isinstance(ma25, pd.Series)
    assert len(ma7) == len(indicators.data)
    assert len(ma25) == len(indicators.data)
    assert not ma7.isnull().any()
    assert not ma25.isnull().any()

def test_macd(indicators):
    """MACD 계산 테스트"""
    macd, signal, hist = indicators.calculate_macd(fast=12, slow=26, signal=9)
    
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert isinstance(hist, pd.Series)
    assert len(macd) == len(indicators.data)
    assert len(signal) == len(indicators.data)
    assert len(hist) == len(indicators.data)
    assert not macd.isnull().any()
    assert not signal.isnull().any()
    assert not hist.isnull().any()

def test_stochastic(indicators):
    """스토캐스틱 계산 테스트"""
    k, d = indicators.calculate_stochastic(k_period=14, d_period=3)
    
    assert isinstance(k, pd.Series)
    assert isinstance(d, pd.Series)
    assert len(k) == len(indicators.data)
    assert len(d) == len(indicators.data)
    assert k.min() >= 0
    assert k.max() <= 100
    assert d.min() >= 0
    assert d.max() <= 100
    assert not k.isnull().any()
    assert not d.isnull().any()

def test_atr(indicators):
    """ATR 계산 테스트"""
    atr = indicators.calculate_atr(period=14)
    
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(indicators.data)
    assert atr.min() >= 0
    assert not atr.isnull().any()

def test_volume_indicators(indicators):
    """거래량 지표 계산 테스트"""
    obv = indicators.calculate_obv()
    vwap = indicators.calculate_vwap()
    
    assert isinstance(obv, pd.Series)
    assert isinstance(vwap, pd.Series)
    assert len(obv) == len(indicators.data)
    assert len(vwap) == len(indicators.data)
    assert not obv.isnull().any()
    assert not vwap.isnull().any()

def test_empty_data():
    """빈 데이터 처리 테스트"""
    empty_data = pd.DataFrame()
    indicators = TechnicalIndicators(empty_data)
    
    with pytest.raises(ValueError):
        indicators.calculate_rsi(period=14)
    
    with pytest.raises(ValueError):
        indicators.calculate_bollinger_bands(period=20, std_dev=2)
    
    with pytest.raises(ValueError):
        indicators.calculate_ma(period=7) 