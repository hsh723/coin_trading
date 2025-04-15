import pytest
from src.model.model_evaluator import ModelEvaluator
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

@pytest.fixture
def model_evaluator():
    config_dir = "./config"
    evaluation_dir = "./evaluation"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "evaluation": {
                "metrics": {
                    "regression": ["mse", "mae", "r2"],
                    "classification": ["accuracy", "precision", "recall", "f1"],
                    "time_series": ["mape", "rmse", "mae"]
                },
                "visualization": {
                    "residual_plot": True,
                    "prediction_plot": True,
                    "error_distribution": True
                },
                "thresholds": {
                    "min_r2": 0.6,
                    "max_mape": 0.2,
                    "min_accuracy": 0.7
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_evaluator.json"), "w") as f:
        json.dump(config, f)
    
    return ModelEvaluator(config_dir=config_dir, evaluation_dir=evaluation_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성 (회귀, 분류, 시계열)
    n_samples = 1000
    
    # 회귀 데이터
    X_reg = np.random.randn(n_samples, 5)
    y_reg = np.random.randn(n_samples)
    
    # 분류 데이터
    X_clf = np.random.randn(n_samples, 5)
    y_clf = np.random.randint(0, 2, n_samples)
    
    # 시계열 데이터
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1H")
    X_ts = pd.DataFrame({
        "price": np.random.normal(100, 10, len(dates)),
        "volume": np.random.normal(1000, 100, len(dates))
    }, index=dates)
    y_ts = np.random.normal(100, 10, len(dates))
    
    return {
        "regression": (X_reg, y_reg),
        "classification": (X_clf, y_clf),
        "time_series": (X_ts, y_ts)
    }

@pytest.fixture
def sample_predictions(sample_data):
    # 샘플 예측값 생성
    predictions = {}
    
    # 회귀 예측
    X_reg, y_reg = sample_data["regression"]
    predictions["regression"] = np.random.randn(len(y_reg))
    
    # 분류 예측
    X_clf, y_clf = sample_data["classification"]
    predictions["classification"] = np.random.randint(0, 2, len(y_clf))
    
    # 시계열 예측
    X_ts, y_ts = sample_data["time_series"]
    predictions["time_series"] = np.random.normal(100, 10, len(y_ts))
    
    return predictions

def test_model_evaluator_initialization(model_evaluator):
    assert model_evaluator is not None
    assert model_evaluator.config_dir == "./config"
    assert model_evaluator.evaluation_dir == "./evaluation"

def test_regression_metrics(model_evaluator, sample_data, sample_predictions):
    # 회귀 메트릭 테스트
    X, y = sample_data["regression"]
    y_pred = sample_predictions["regression"]
    metrics = model_evaluator.evaluate_regression(y, y_pred)
    
    assert metrics is not None
    assert "mse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    
    # 메트릭 값 범위 확인
    assert metrics["mse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["r2"] <= 1

def test_classification_metrics(model_evaluator, sample_data, sample_predictions):
    # 분류 메트릭 테스트
    X, y = sample_data["classification"]
    y_pred = sample_predictions["classification"]
    metrics = model_evaluator.evaluate_classification(y, y_pred)
    
    assert metrics is not None
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    
    # 메트릭 값 범위 확인
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1

def test_time_series_metrics(model_evaluator, sample_data, sample_predictions):
    # 시계열 메트릭 테스트
    X, y = sample_data["time_series"]
    y_pred = sample_predictions["time_series"]
    metrics = model_evaluator.evaluate_time_series(y, y_pred)
    
    assert metrics is not None
    assert "mape" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    
    # 메트릭 값 범위 확인
    assert metrics["mape"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0

def test_residual_plot(model_evaluator, sample_data, sample_predictions):
    # 잔차 플롯 테스트
    X, y = sample_data["regression"]
    y_pred = sample_predictions["regression"]
    fig = model_evaluator.plot_residuals(y, y_pred)
    
    assert fig is not None

def test_prediction_plot(model_evaluator, sample_data, sample_predictions):
    # 예측 플롯 테스트
    X, y = sample_data["time_series"]
    y_pred = sample_predictions["time_series"]
    fig = model_evaluator.plot_predictions(y, y_pred)
    
    assert fig is not None

def test_error_distribution(model_evaluator, sample_data, sample_predictions):
    # 오차 분포 테스트
    X, y = sample_data["regression"]
    y_pred = sample_predictions["regression"]
    fig = model_evaluator.plot_error_distribution(y, y_pred)
    
    assert fig is not None

def test_error_handling(model_evaluator):
    # 에러 처리 테스트
    # 잘못된 데이터
    with pytest.raises(ValueError):
        model_evaluator.evaluate_regression(None, None)
    
    # 잘못된 예측값
    with pytest.raises(ValueError):
        model_evaluator.evaluate_classification(np.array([1]), np.array([1, 2]))
    
    # 잘못된 메트릭
    with pytest.raises(ValueError):
        model_evaluator.evaluate_time_series(np.array([1]), np.array([1]), metrics=["invalid"])

def test_evaluation_performance(model_evaluator, sample_data, sample_predictions):
    # 평가 성능 테스트
    X, y = sample_data["regression"]
    y_pred = sample_predictions["regression"]
    start_time = datetime.now()
    
    # 회귀 평가 실행
    model_evaluator.evaluate_regression(y, y_pred)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1,000개 샘플에 대한 평가를 1초 이내에 완료
    assert processing_time < 1.0

def test_evaluation_configuration(model_evaluator):
    # 평가 설정 테스트
    config = model_evaluator.get_configuration()
    
    assert config is not None
    assert "evaluation" in config
    assert "metrics" in config["evaluation"]
    assert "visualization" in config["evaluation"]
    assert "thresholds" in config["evaluation"]
    assert "regression" in config["evaluation"]["metrics"]
    assert "classification" in config["evaluation"]["metrics"]
    assert "time_series" in config["evaluation"]["metrics"]
    assert "residual_plot" in config["evaluation"]["visualization"]
    assert "prediction_plot" in config["evaluation"]["visualization"]
    assert "error_distribution" in config["evaluation"]["visualization"]
    assert "min_r2" in config["evaluation"]["thresholds"]
    assert "max_mape" in config["evaluation"]["thresholds"]
    assert "min_accuracy" in config["evaluation"]["thresholds"] 