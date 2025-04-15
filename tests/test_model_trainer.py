import pytest
from src.model.model_trainer import ModelTrainer
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

@pytest.fixture
def model_trainer():
    config_dir = "./config"
    model_dir = "./models"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "training": {
                "model_type": "regression",
                "features": ["price", "volume", "rsi", "macd"],
                "target": "next_price",
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "batch_size": 32,
                "epochs": 10,
                "learning_rate": 0.001,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_trainer.json"), "w") as f:
        json.dump(config, f)
    
    return ModelTrainer(config_dir=config_dir, model_dir=model_dir, data_dir=data_dir)

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

def test_model_trainer_initialization(model_trainer):
    assert model_trainer is not None
    assert model_trainer.config_dir == "./config"
    assert model_trainer.model_dir == "./models"
    assert model_trainer.data_dir == "./data"

def test_data_preprocessing(model_trainer, sample_data):
    # 데이터 전처리 테스트
    processed_data = model_trainer.preprocess_data(sample_data)
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

def test_data_splitting(model_trainer, sample_data):
    # 데이터 분할 테스트
    train_data, val_data, test_data = model_trainer.split_data(sample_data)
    
    # 데이터 크기 확인
    total_size = len(sample_data)
    assert len(train_data) == int(total_size * 0.7)
    assert len(val_data) == int(total_size * 0.15)
    assert len(test_data) == int(total_size * 0.15)
    
    # 데이터 인덱스 확인
    assert all(train_data.index < val_data.index[0])
    assert all(val_data.index < test_data.index[0])

def test_model_creation(model_trainer):
    # 모델 생성 테스트
    model = model_trainer.create_model()
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

def test_model_training(model_trainer, sample_data):
    # 모델 학습 테스트
    history = model_trainer.train_model(sample_data)
    assert history is not None
    assert "loss" in history.history
    assert "val_loss" in history.history
    assert len(history.history["loss"]) == 10  # epochs=10

def test_model_evaluation(model_trainer, sample_data):
    # 모델 평가 테스트
    metrics = model_trainer.evaluate_model(sample_data)
    assert metrics is not None
    assert "mse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

def test_model_saving(model_trainer):
    # 모델 저장 테스트
    model_path = os.path.join(model_trainer.model_dir, "test_model.h5")
    model_trainer.save_model(model_path)
    assert os.path.exists(model_path)

def test_model_loading(model_trainer):
    # 모델 로딩 테스트
    model_path = os.path.join(model_trainer.model_dir, "test_model.h5")
    model = model_trainer.load_model(model_path)
    assert model is not None
    assert hasattr(model, "predict")

def test_hyperparameter_tuning(model_trainer, sample_data):
    # 하이퍼파라미터 튜닝 테스트
    best_params = model_trainer.tune_hyperparameters(sample_data)
    assert best_params is not None
    assert "learning_rate" in best_params
    assert "batch_size" in best_params
    assert "epochs" in best_params

def test_early_stopping(model_trainer, sample_data):
    # 조기 종료 테스트
    history = model_trainer.train_model(sample_data, early_stopping=True)
    assert history is not None
    assert len(history.history["loss"]) <= 10  # 최대 epochs=10

def test_error_handling(model_trainer):
    # 에러 처리 테스트
    # 잘못된 데이터
    with pytest.raises(ValueError):
        model_trainer.train_model(None)
    
    # 잘못된 모델 타입
    with pytest.raises(ValueError):
        model_trainer.create_model(model_type="invalid_type")
    
    # 잘못된 데이터 분할 비율
    with pytest.raises(ValueError):
        model_trainer.split_data(sample_data, train_ratio=0.8, val_ratio=0.3)

def test_training_performance(model_trainer, sample_data):
    # 학습 성능 테스트
    start_time = datetime.now()
    
    # 모델 학습 실행
    model_trainer.train_model(sample_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1,000개 샘플에 대한 학습을 30초 이내에 완료
    assert processing_time < 30.0

def test_training_configuration(model_trainer):
    # 학습 설정 테스트
    config = model_trainer.get_configuration()
    
    assert config is not None
    assert "training" in config
    assert "model_type" in config["training"]
    assert "features" in config["training"]
    assert "target" in config["training"]
    assert "train_ratio" in config["training"]
    assert "val_ratio" in config["training"]
    assert "test_ratio" in config["training"]
    assert "batch_size" in config["training"]
    assert "epochs" in config["training"]
    assert "learning_rate" in config["training"]
    assert "early_stopping" in config["training"]

def test_model_interpretation(model_trainer, sample_data):
    # 모델 해석 테스트
    X, y = sample_data
    model = model_trainer.train_model(X, y)
    interpretation = model_trainer.interpret_model(model, X)
    assert interpretation is not None
    assert "shap_values" in interpretation
    assert "feature_importance" in interpretation

def test_model_performance(model_trainer, sample_data):
    # 모델 성능 테스트
    X, y = sample_data
    start_time = datetime.now()
    
    # 모델 학습 및 예측
    model = model_trainer.train_model(X, y)
    predictions = model.predict(X)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 10,000개 샘플을 5초 이내에 처리
    assert processing_time < 5.0
    
    # 예측 정확도 확인
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.5  # 랜덤 예측보다는 좋아야 함 