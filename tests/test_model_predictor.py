import pytest
from src.model.model_predictor import ModelPredictor
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_predictor():
    config_dir = "./config"
    model_dir = "./models"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "prediction": {
                "model_type": "regression",
                "features": ["price", "volume", "rsi", "macd"],
                "target": "next_price",
                "prediction_horizon": 1,
                "confidence_threshold": 0.8,
                "batch_size": 32,
                "sequence_length": 60
            }
        }
    }
    with open(os.path.join(config_dir, "model_predictor.json"), "w") as f:
        json.dump(config, f)
    
    return ModelPredictor(config_dir=config_dir, model_dir=model_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1H")
    data = {
        "price": np.random.normal(100, 10, len(dates)),
        "volume": np.random.normal(1000, 100, len(dates)),
        "rsi": np.random.uniform(0, 100, len(dates)),
        "macd": np.random.normal(0, 1, len(dates)),
        "next_price": np.random.normal(100, 10, len(dates))
    }
    return pd.DataFrame(data, index=dates)

def test_model_predictor_initialization(model_predictor):
    assert model_predictor is not None
    assert model_predictor.config_dir == "./config"
    assert model_predictor.model_dir == "./models"

def test_data_preprocessing(model_predictor, sample_data):
    # 데이터 전처리 테스트
    processed_data = model_predictor.preprocess_data(sample_data)
    assert processed_data is not None
    assert isinstance(processed_data, tuple)
    assert len(processed_data) == 2  # X, y
    
    # 특성 확인
    X, y = processed_data
    assert X.shape[1] == 4  # features 개수
    assert "price" in X.columns
    assert "volume" in X.columns
    assert "rsi" in X.columns
    assert "macd" in X.columns

def test_sequence_generation(model_predictor, sample_data):
    # 시퀀스 생성 테스트
    sequences = model_predictor.generate_sequences(sample_data)
    assert sequences is not None
    assert isinstance(sequences, np.ndarray)
    assert len(sequences.shape) == 3  # (samples, sequence_length, features)
    assert sequences.shape[1] == 60  # sequence_length
    assert sequences.shape[2] == 4  # features

def test_single_prediction(model_predictor, sample_data):
    # 단일 예측 테스트
    prediction = model_predictor.predict_single(sample_data.iloc[0])
    assert prediction is not None
    assert "value" in prediction
    assert "confidence" in prediction
    assert 0 <= prediction["confidence"] <= 1

def test_batch_prediction(model_predictor, sample_data):
    # 배치 예측 테스트
    predictions = model_predictor.predict_batch(sample_data)
    assert predictions is not None
    assert isinstance(predictions, pd.DataFrame)
    assert "prediction" in predictions.columns
    assert "confidence" in predictions.columns
    assert len(predictions) == len(sample_data)

def test_multi_step_prediction(model_predictor, sample_data):
    # 다중 스텝 예측 테스트
    predictions = model_predictor.predict_multi_step(sample_data, steps=5)
    assert predictions is not None
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 5  # steps=5
    assert all(col.startswith("step_") for col in predictions.columns)

def test_confidence_threshold(model_predictor, sample_data):
    # 신뢰도 임계값 테스트
    predictions = model_predictor.predict_batch(sample_data, confidence_threshold=0.9)
    assert all(predictions["confidence"] >= 0.9)

def test_prediction_visualization(model_predictor, sample_data):
    # 예측 시각화 테스트
    predictions = model_predictor.predict_batch(sample_data)
    fig = model_predictor.plot_predictions(sample_data, predictions)
    assert fig is not None

def test_error_handling(model_predictor):
    # 에러 처리 테스트
    # 잘못된 데이터
    with pytest.raises(ValueError):
        model_predictor.predict_single(None)
    
    # 잘못된 배치 크기
    with pytest.raises(ValueError):
        model_predictor.predict_batch(sample_data, batch_size=0)
    
    # 잘못된 예측 기간
    with pytest.raises(ValueError):
        model_predictor.predict_multi_step(sample_data, steps=0)

def test_prediction_performance(model_predictor, sample_data):
    # 예측 성능 테스트
    start_time = datetime.now()
    
    # 배치 예측 실행
    model_predictor.predict_batch(sample_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1,000개 샘플에 대한 예측을 1초 이내에 완료
    assert processing_time < 1.0

def test_prediction_configuration(model_predictor):
    # 예측 설정 테스트
    config = model_predictor.get_configuration()
    
    assert config is not None
    assert "prediction" in config
    assert "model_type" in config["prediction"]
    assert "features" in config["prediction"]
    assert "target" in config["prediction"]
    assert "prediction_horizon" in config["prediction"]
    assert "confidence_threshold" in config["prediction"]
    assert "batch_size" in config["prediction"]
    assert "sequence_length" in config["prediction"] 