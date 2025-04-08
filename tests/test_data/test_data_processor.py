import pytest
import pandas as pd
import numpy as np
from src.data.processor import DataProcessor

def test_data_processor_initialization():
    """데이터 처리기 초기화 테스트"""
    processor = DataProcessor()
    assert processor is not None
    assert processor.logger is not None

def test_preprocess_data(sample_market_data):
    """데이터 전처리 테스트"""
    processor = DataProcessor()
    
    # 결측치 처리 테스트
    data_with_nan = sample_market_data.copy()
    data_with_nan.iloc[0, 0] = np.nan
    processed_data = processor.preprocess_data(data_with_nan)
    assert not processed_data.isnull().any().any()
    
    # 이상치 처리 테스트
    data_with_outlier = sample_market_data.copy()
    data_with_outlier.iloc[0, 0] = 1000  # 이상치 추가
    processed_data = processor.preprocess_data(data_with_outlier)
    assert processed_data.iloc[0, 0] != 1000
    
    # 정규화 테스트
    normalized_data = processor.preprocess_data(sample_market_data, normalize=True)
    assert normalized_data['close'].mean() == pytest.approx(0, abs=1e-6)
    assert normalized_data['close'].std() == pytest.approx(1, abs=1e-6)

def test_add_technical_indicators(sample_market_data):
    """기술적 지표 추가 테스트"""
    processor = DataProcessor()
    
    # 기본 지표 추가
    data_with_indicators = processor.add_technical_indicators(sample_market_data)
    assert 'rsi' in data_with_indicators.columns
    assert 'macd' in data_with_indicators.columns
    assert 'signal' in data_with_indicators.columns
    assert 'bb_upper' in data_with_indicators.columns
    assert 'bb_middle' in data_with_indicators.columns
    assert 'bb_lower' in data_with_indicators.columns
    
    # 사용자 지정 지표 추가
    custom_indicators = {
        'sma': {'period': 20},
        'ema': {'period': 10}
    }
    data_with_custom = processor.add_technical_indicators(
        sample_market_data,
        indicators=custom_indicators
    )
    assert 'sma_20' in data_with_custom.columns
    assert 'ema_10' in data_with_custom.columns

def test_resample_timeframe(sample_market_data):
    """시간대 리샘플링 테스트"""
    processor = DataProcessor()
    
    # 1시간에서 4시간으로 리샘플링
    resampled_data = processor.resample_timeframe(
        sample_market_data,
        source_timeframe='1h',
        target_timeframe='4h'
    )
    
    assert len(resampled_data) <= len(sample_market_data)
    assert resampled_data.index.freq == '4H'
    
    # OHLCV 값 검증
    assert resampled_data['open'].iloc[0] == sample_market_data['open'].iloc[0]
    assert resampled_data['high'].iloc[0] >= sample_market_data['high'].iloc[0:4].max()
    assert resampled_data['low'].iloc[0] <= sample_market_data['low'].iloc[0:4].min()
    assert resampled_data['close'].iloc[0] == sample_market_data['close'].iloc[3]
    assert resampled_data['volume'].iloc[0] == sample_market_data['volume'].iloc[0:4].sum()

def test_prepare_dataset(sample_market_data):
    """데이터셋 준비 테스트"""
    processor = DataProcessor()
    
    # 기본 분할 비율로 데이터셋 준비
    train_data, val_data, test_data = processor.prepare_dataset(sample_market_data)
    
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0
    assert len(train_data) + len(val_data) + len(test_data) == len(sample_market_data)
    
    # 사용자 지정 분할 비율로 데이터셋 준비
    train_data, val_data, test_data = processor.prepare_dataset(
        sample_market_data,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    assert len(train_data) / len(sample_market_data) == pytest.approx(0.6, abs=0.01)
    assert len(val_data) / len(sample_market_data) == pytest.approx(0.2, abs=0.01)
    assert len(test_data) / len(sample_market_data) == pytest.approx(0.2, abs=0.01)

def test_handle_missing_values():
    """결측치 처리 테스트"""
    processor = DataProcessor()
    
    # 결측치가 있는 데이터 생성
    data = pd.DataFrame({
        'close': [1, 2, np.nan, 4, 5],
        'volume': [100, np.nan, 300, 400, 500]
    })
    
    # 전진 채우기
    filled_data = processor._handle_missing_values(data, method='ffill')
    assert not filled_data.isnull().any().any()
    assert filled_data['close'].iloc[2] == 2
    
    # 후진 채우기
    filled_data = processor._handle_missing_values(data, method='bfill')
    assert not filled_data.isnull().any().any()
    assert filled_data['close'].iloc[2] == 4
    
    # 평균값으로 채우기
    filled_data = processor._handle_missing_values(data, method='mean')
    assert not filled_data.isnull().any().any()
    assert filled_data['close'].iloc[2] == pytest.approx(3, abs=0.1)

def test_handle_outliers(sample_market_data):
    """이상치 처리 테스트"""
    processor = DataProcessor()
    
    # 이상치가 있는 데이터 생성
    data = sample_market_data.copy()
    data.iloc[0, 0] = 1000  # 이상치 추가
    
    # Z-score 기반 이상치 처리
    processed_data = processor._handle_outliers(data, method='zscore', threshold=3)
    assert processed_data.iloc[0, 0] != 1000
    
    # IQR 기반 이상치 처리
    processed_data = processor._handle_outliers(data, method='iqr', threshold=1.5)
    assert processed_data.iloc[0, 0] != 1000

def test_normalize_data(sample_market_data):
    """데이터 정규화 테스트"""
    processor = DataProcessor()
    
    # Z-score 정규화
    normalized_data = processor._normalize_data(sample_market_data, method='zscore')
    assert normalized_data['close'].mean() == pytest.approx(0, abs=1e-6)
    assert normalized_data['close'].std() == pytest.approx(1, abs=1e-6)
    
    # Min-Max 정규화
    normalized_data = processor._normalize_data(sample_market_data, method='minmax')
    assert normalized_data['close'].min() == pytest.approx(0, abs=1e-6)
    assert normalized_data['close'].max() == pytest.approx(1, abs=1e-6) 