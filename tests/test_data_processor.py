import pytest
from src.data.data_processor import DataProcessor
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def data_processor():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "data_types": ["trades", "orderbook", "ohlcv"],
            "processing": {
                "resample_freq": "1m",
                "fill_method": "ffill",
                "normalization": True
            },
            "features": {
                "technical_indicators": ["sma", "ema", "rsi", "macd"],
                "statistical_indicators": ["zscore", "volatility"],
                "volume_indicators": ["obv", "vwap"]
            }
        }
    }
    with open(os.path.join(config_dir, "data_processor.json"), "w") as f:
        json.dump(config, f)
    
    return DataProcessor(config_dir=config_dir, data_dir=data_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="1min")
    data = pd.DataFrame({
        "open": np.random.normal(50000, 1000, len(dates)),
        "high": np.random.normal(51000, 1000, len(dates)),
        "low": np.random.normal(49000, 1000, len(dates)),
        "close": np.random.normal(50500, 1000, len(dates)),
        "volume": np.random.normal(100, 10, len(dates))
    }, index=dates)
    return data

def test_data_processor_initialization(data_processor):
    assert data_processor is not None
    assert data_processor.config_dir == "./config"
    assert data_processor.data_dir == "./data"

def test_data_loading(data_processor, sample_data):
    # 데이터 로드 테스트
    loaded_data = data_processor.load_data("sample_data.csv")
    assert loaded_data is not None
    assert isinstance(loaded_data, pd.DataFrame)

def test_data_resampling(data_processor, sample_data):
    # 데이터 리샘플링 테스트
    resampled_data = data_processor.resample_data(sample_data, "5min")
    assert resampled_data is not None
    assert len(resampled_data) < len(sample_data)
    assert resampled_data.index.freq == "5min"

def test_data_cleaning(data_processor, sample_data):
    # 데이터 클리닝 테스트
    # 일부 데이터에 결측치 추가
    dirty_data = sample_data.copy()
    dirty_data.iloc[10:20] = np.nan
    
    cleaned_data = data_processor.clean_data(dirty_data)
    assert cleaned_data is not None
    assert cleaned_data.isna().sum().sum() == 0

def test_feature_engineering(data_processor, sample_data):
    # 특성 공학 테스트
    features = data_processor.generate_features(sample_data)
    assert features is not None
    assert "sma_20" in features.columns
    assert "rsi_14" in features.columns
    assert "macd" in features.columns

def test_data_normalization(data_processor, sample_data):
    # 데이터 정규화 테스트
    normalized_data = data_processor.normalize_data(sample_data)
    assert normalized_data is not None
    assert np.all(normalized_data.mean() < 1.0)
    assert np.all(normalized_data.std() < 1.0)

def test_data_splitting(data_processor, sample_data):
    # 데이터 분할 테스트
    train_data, test_data = data_processor.split_data(sample_data, test_size=0.2)
    assert train_data is not None
    assert test_data is not None
    assert len(train_data) + len(test_data) == len(sample_data)

def test_data_augmentation(data_processor, sample_data):
    # 데이터 증강 테스트
    augmented_data = data_processor.augment_data(sample_data)
    assert augmented_data is not None
    assert len(augmented_data) > len(sample_data)

def test_data_validation(data_processor, sample_data):
    # 데이터 유효성 검사 테스트
    assert data_processor.validate_data(sample_data) is True
    
    # 잘못된 데이터 생성
    invalid_data = sample_data.copy()
    invalid_data["close"] = -invalid_data["close"]
    assert data_processor.validate_data(invalid_data) is False

def test_data_export(data_processor, sample_data):
    # 데이터 내보내기 테스트
    export_path = os.path.join(data_processor.data_dir, "exported_data.csv")
    data_processor.export_data(sample_data, export_path)
    assert os.path.exists(export_path)
    
    # 내보낸 데이터 확인
    exported_data = pd.read_csv(export_path)
    assert len(exported_data) == len(sample_data)

def test_data_processing_pipeline(data_processor, sample_data):
    # 데이터 처리 파이프라인 테스트
    processed_data = data_processor.process_data(sample_data)
    assert processed_data is not None
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) > 0
    
    # 처리된 데이터의 특성 확인
    assert "sma_20" in processed_data.columns
    assert "rsi_14" in processed_data.columns
    assert "macd" in processed_data.columns
    assert processed_data.isna().sum().sum() == 0

def test_data_processor_performance(data_processor, sample_data):
    # 데이터 처리기 성능 테스트
    start_time = datetime.now()
    
    # 대량의 데이터 처리
    large_data = pd.concat([sample_data] * 10)
    processed_data = data_processor.process_data(large_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100,000개 행의 데이터를 10초 이내에 처리
    assert processing_time < 10.0
    assert len(processed_data) == len(large_data) 