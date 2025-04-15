import pytest
from src.model.model_optimizer import ModelOptimizer
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

@pytest.fixture
def model_optimizer():
    config_dir = "./config"
    model_dir = "./models"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "optimization": {
                "method": "bayesian",
                "hyperparameters": {
                    "learning_rate": {"min": 0.0001, "max": 0.1},
                    "batch_size": {"min": 16, "max": 256},
                    "hidden_units": {"min": 32, "max": 512},
                    "dropout_rate": {"min": 0.0, "max": 0.5}
                },
                "n_trials": 50,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                },
                "objective": "minimize",
                "metric": "val_loss"
            }
        }
    }
    with open(os.path.join(config_dir, "model_optimizer.json"), "w") as f:
        json.dump(config, f)
    
    return ModelOptimizer(config_dir=config_dir, model_dir=model_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    n_samples = 1000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return X, y

def test_model_optimizer_initialization(model_optimizer):
    assert model_optimizer is not None
    assert model_optimizer.config_dir == "./config"
    assert model_optimizer.model_dir == "./models"

def test_hyperparameter_space(model_optimizer):
    # 하이퍼파라미터 공간 테스트
    space = model_optimizer.get_hyperparameter_space()
    
    assert space is not None
    assert "learning_rate" in space
    assert "batch_size" in space
    assert "hidden_units" in space
    assert "dropout_rate" in space
    
    # 범위 확인
    assert space["learning_rate"]["min"] == 0.0001
    assert space["learning_rate"]["max"] == 0.1
    assert space["batch_size"]["min"] == 16
    assert space["batch_size"]["max"] == 256
    assert space["hidden_units"]["min"] == 32
    assert space["hidden_units"]["max"] == 512
    assert space["dropout_rate"]["min"] == 0.0
    assert space["dropout_rate"]["max"] == 0.5

def test_optimization_method(model_optimizer, sample_data):
    # 최적화 방법 테스트
    X, y = sample_data
    best_params = model_optimizer.optimize_hyperparameters(X, y)
    
    assert best_params is not None
    assert "learning_rate" in best_params
    assert "batch_size" in best_params
    assert "hidden_units" in best_params
    assert "dropout_rate" in best_params
    
    # 파라미터 범위 확인
    assert 0.0001 <= best_params["learning_rate"] <= 0.1
    assert 16 <= best_params["batch_size"] <= 256
    assert 32 <= best_params["hidden_units"] <= 512
    assert 0.0 <= best_params["dropout_rate"] <= 0.5

def test_early_stopping(model_optimizer, sample_data):
    # 조기 종료 테스트
    X, y = sample_data
    history = model_optimizer.optimize_hyperparameters(X, y, early_stopping=True)
    
    assert history is not None
    assert "trials" in history
    assert "best_params" in history
    assert "best_score" in history
    
    # 조기 종료 확인
    assert len(history["trials"]) <= 50  # 최대 시도 횟수

def test_objective_function(model_optimizer, sample_data):
    # 목적 함수 테스트
    X, y = sample_data
    score = model_optimizer.evaluate_objective(X, y)
    
    assert score is not None
    assert isinstance(score, float)

def test_parameter_constraints(model_optimizer):
    # 파라미터 제약 조건 테스트
    constraints = model_optimizer.get_parameter_constraints()
    
    assert constraints is not None
    assert "learning_rate" in constraints
    assert "batch_size" in constraints
    assert "hidden_units" in constraints
    assert "dropout_rate" in constraints

def test_optimization_history(model_optimizer, sample_data):
    # 최적화 기록 테스트
    X, y = sample_data
    history = model_optimizer.optimize_hyperparameters(X, y)
    
    assert history is not None
    assert "trials" in history
    assert "scores" in history
    assert "params" in history
    assert len(history["trials"]) == 50  # n_trials=50

def test_error_handling(model_optimizer):
    # 에러 처리 테스트
    # 잘못된 데이터
    with pytest.raises(ValueError):
        model_optimizer.optimize_hyperparameters(None, None)
    
    # 잘못된 최적화 방법
    with pytest.raises(ValueError):
        model_optimizer.set_optimization_method("invalid_method")
    
    # 잘못된 목적 함수
    with pytest.raises(ValueError):
        model_optimizer.set_objective("invalid_objective")

def test_optimization_performance(model_optimizer, sample_data):
    # 최적화 성능 테스트
    X, y = sample_data
    start_time = datetime.now()
    
    # 하이퍼파라미터 최적화 실행
    model_optimizer.optimize_hyperparameters(X, y)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 50회 시도에 대한 최적화를 300초 이내에 완료
    assert processing_time < 300.0

def test_optimization_configuration(model_optimizer):
    # 최적화 설정 테스트
    config = model_optimizer.get_configuration()
    
    assert config is not None
    assert "optimization" in config
    assert "method" in config["optimization"]
    assert "hyperparameters" in config["optimization"]
    assert "n_trials" in config["optimization"]
    assert "early_stopping" in config["optimization"]
    assert "objective" in config["optimization"]
    assert "metric" in config["optimization"]
    assert "learning_rate" in config["optimization"]["hyperparameters"]
    assert "batch_size" in config["optimization"]["hyperparameters"]
    assert "hidden_units" in config["optimization"]["hyperparameters"]
    assert "dropout_rate" in config["optimization"]["hyperparameters"] 