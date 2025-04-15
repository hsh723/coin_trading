import pytest
from src.model.model_ensemble import ModelEnsemble
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_ensemble():
    config_dir = "./config"
    model_dir = "./models"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "ensemble": {
                "methods": ["voting", "bagging", "boosting", "stacking"],
                "base_models": [
                    {"type": "regression", "name": "linear"},
                    {"type": "regression", "name": "random_forest"},
                    {"type": "regression", "name": "xgboost"}
                ],
                "weights": {
                    "linear": 0.3,
                    "random_forest": 0.4,
                    "xgboost": 0.3
                },
                "voting": {
                    "type": "soft",
                    "weights": [0.3, 0.4, 0.3]
                },
                "bagging": {
                    "n_estimators": 10,
                    "max_samples": 0.8,
                    "max_features": 0.8
                },
                "boosting": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3
                },
                "stacking": {
                    "n_folds": 5,
                    "final_estimator": "linear"
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_ensemble.json"), "w") as f:
        json.dump(config, f)
    
    return ModelEnsemble(config_dir=config_dir, model_dir=model_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    n_samples = 1000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y

def test_model_ensemble_initialization(model_ensemble):
    assert model_ensemble is not None
    assert model_ensemble.config_dir == "./config"
    assert model_ensemble.model_dir == "./models"

def test_base_models_creation(model_ensemble):
    # 기본 모델 생성 테스트
    base_models = model_ensemble.create_base_models()
    
    assert base_models is not None
    assert len(base_models) == 3  # 3개의 기본 모델
    assert all(hasattr(model, "fit") for model in base_models)
    assert all(hasattr(model, "predict") for model in base_models)

def test_voting_ensemble(model_ensemble, sample_data):
    # 투표 앙상블 테스트
    X, y = sample_data
    voting_model = model_ensemble.create_voting_ensemble()
    
    assert voting_model is not None
    assert hasattr(voting_model, "fit")
    assert hasattr(voting_model, "predict")
    
    # 모델 학습 및 예측
    voting_model.fit(X, y)
    predictions = voting_model.predict(X)
    assert len(predictions) == len(y)

def test_bagging_ensemble(model_ensemble, sample_data):
    # 배깅 앙상블 테스트
    X, y = sample_data
    bagging_model = model_ensemble.create_bagging_ensemble()
    
    assert bagging_model is not None
    assert hasattr(bagging_model, "fit")
    assert hasattr(bagging_model, "predict")
    
    # 모델 학습 및 예측
    bagging_model.fit(X, y)
    predictions = bagging_model.predict(X)
    assert len(predictions) == len(y)

def test_boosting_ensemble(model_ensemble, sample_data):
    # 부스팅 앙상블 테스트
    X, y = sample_data
    boosting_model = model_ensemble.create_boosting_ensemble()
    
    assert boosting_model is not None
    assert hasattr(boosting_model, "fit")
    assert hasattr(boosting_model, "predict")
    
    # 모델 학습 및 예측
    boosting_model.fit(X, y)
    predictions = boosting_model.predict(X)
    assert len(predictions) == len(y)

def test_stacking_ensemble(model_ensemble, sample_data):
    # 스태킹 앙상블 테스트
    X, y = sample_data
    stacking_model = model_ensemble.create_stacking_ensemble()
    
    assert stacking_model is not None
    assert hasattr(stacking_model, "fit")
    assert hasattr(stacking_model, "predict")
    
    # 모델 학습 및 예측
    stacking_model.fit(X, y)
    predictions = stacking_model.predict(X)
    assert len(predictions) == len(y)

def test_ensemble_weights(model_ensemble):
    # 앙상블 가중치 테스트
    weights = model_ensemble.get_ensemble_weights()
    
    assert weights is not None
    assert "linear" in weights
    assert "random_forest" in weights
    assert "xgboost" in weights
    assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

def test_ensemble_performance(model_ensemble, sample_data):
    # 앙상블 성능 테스트
    X, y = sample_data
    start_time = datetime.now()
    
    # 모든 앙상블 모델 생성 및 학습
    voting_model = model_ensemble.create_voting_ensemble()
    bagging_model = model_ensemble.create_bagging_ensemble()
    boosting_model = model_ensemble.create_boosting_ensemble()
    stacking_model = model_ensemble.create_stacking_ensemble()
    
    voting_model.fit(X, y)
    bagging_model.fit(X, y)
    boosting_model.fit(X, y)
    stacking_model.fit(X, y)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1,000개 샘플에 대한 모든 앙상블 학습을 60초 이내에 완료
    assert processing_time < 60.0

def test_error_handling(model_ensemble):
    # 에러 처리 테스트
    # 잘못된 앙상블 방법
    with pytest.raises(ValueError):
        model_ensemble.create_ensemble("invalid_method")
    
    # 잘못된 가중치
    with pytest.raises(ValueError):
        model_ensemble.set_ensemble_weights({"invalid_model": 1.0})
    
    # 잘못된 기본 모델
    with pytest.raises(ValueError):
        model_ensemble.add_base_model("invalid_type", "invalid_name")

def test_ensemble_configuration(model_ensemble):
    # 앙상블 설정 테스트
    config = model_ensemble.get_configuration()
    
    assert config is not None
    assert "ensemble" in config
    assert "methods" in config["ensemble"]
    assert "base_models" in config["ensemble"]
    assert "weights" in config["ensemble"]
    assert "voting" in config["ensemble"]
    assert "bagging" in config["ensemble"]
    assert "boosting" in config["ensemble"]
    assert "stacking" in config["ensemble"]
    assert "type" in config["ensemble"]["voting"]
    assert "n_estimators" in config["ensemble"]["bagging"]
    assert "learning_rate" in config["ensemble"]["boosting"]
    assert "n_folds" in config["ensemble"]["stacking"] 