import pytest
from src.model.model_validator import ModelValidator
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@pytest.fixture
def model_validator():
    config_dir = "./config"
    validation_dir = "./validation"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "validation": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "cross_validation": {
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 42
                },
                "thresholds": {
                    "min_accuracy": 0.7,
                    "min_precision": 0.6,
                    "min_recall": 0.6,
                    "min_f1": 0.6
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_validator.json"), "w") as f:
        json.dump(config, f)
    
    return ModelValidator(config_dir=config_dir, validation_dir=validation_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성 (분류 문제)
    n_samples = 1000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y

@pytest.fixture
def sample_model():
    # 샘플 모델 클래스 생성
    class DummyModel:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        
        def predict_proba(self, X):
            return np.random.rand(len(X), 2)
    
    return DummyModel()

def test_model_validator_initialization(model_validator):
    assert model_validator is not None
    assert model_validator.config_dir == "./config"
    assert model_validator.validation_dir == "./validation"

def test_metrics_calculation(model_validator, sample_data, sample_model):
    # 메트릭 계산 테스트
    X, y = sample_data
    metrics = model_validator.calculate_metrics(sample_model, X, y)
    
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

def test_cross_validation(model_validator, sample_data, sample_model):
    # 교차 검증 테스트
    X, y = sample_data
    cv_results = model_validator.cross_validate(sample_model, X, y)
    
    assert cv_results is not None
    assert "scores" in cv_results
    assert "mean" in cv_results
    assert "std" in cv_results
    
    # 5-fold 교차 검증 결과 확인
    assert len(cv_results["scores"]) == 5

def test_threshold_validation(model_validator, sample_data, sample_model):
    # 임계값 검증 테스트
    X, y = sample_data
    validation_result = model_validator.validate_thresholds(sample_model, X, y)
    
    assert validation_result is not None
    assert isinstance(validation_result, dict)
    assert all(isinstance(value, bool) for value in validation_result.values())

def test_feature_importance(model_validator, sample_data, sample_model):
    # 특성 중요도 테스트
    X, y = sample_data
    importance = model_validator.calculate_feature_importance(sample_model, X, y)
    
    assert importance is not None
    assert isinstance(importance, np.ndarray)
    assert len(importance) == X.shape[1]  # 특성 개수와 동일

def test_confusion_matrix(model_validator, sample_data, sample_model):
    # 혼동 행렬 테스트
    X, y = sample_data
    cm = model_validator.calculate_confusion_matrix(sample_model, X, y)
    
    assert cm is not None
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (2, 2)  # 이진 분류의 경우 2x2 행렬

def test_roc_curve(model_validator, sample_data, sample_model):
    # ROC 곡선 테스트
    X, y = sample_data
    roc = model_validator.calculate_roc_curve(sample_model, X, y)
    
    assert roc is not None
    assert "fpr" in roc
    assert "tpr" in roc
    assert "auc" in roc
    assert 0 <= roc["auc"] <= 1

def test_precision_recall_curve(model_validator, sample_data, sample_model):
    # 정밀도-재현율 곡선 테스트
    X, y = sample_data
    pr = model_validator.calculate_precision_recall_curve(sample_model, X, y)
    
    assert pr is not None
    assert "precision" in pr
    assert "recall" in pr
    assert "ap" in pr
    assert 0 <= pr["ap"] <= 1

def test_error_handling(model_validator):
    # 에러 처리 테스트
    # 잘못된 데이터
    with pytest.raises(ValueError):
        model_validator.calculate_metrics(None, None, None)
    
    # 잘못된 모델
    with pytest.raises(ValueError):
        model_validator.cross_validate(None, np.random.rand(10, 5), np.random.randint(0, 2, 10))
    
    # 잘못된 임계값
    with pytest.raises(ValueError):
        model_validator.validate_thresholds(None, None, None, min_accuracy=1.5)

def test_validation_performance(model_validator, sample_data, sample_model):
    # 검증 성능 테스트
    X, y = sample_data
    start_time = datetime.now()
    
    # 교차 검증 실행
    model_validator.cross_validate(sample_model, X, y)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 5-fold 교차 검증을 30초 이내에 완료
    assert processing_time < 30.0

def test_validation_configuration(model_validator):
    # 검증 설정 테스트
    config = model_validator.get_configuration()
    
    assert config is not None
    assert "validation" in config
    assert "metrics" in config["validation"]
    assert "cross_validation" in config["validation"]
    assert "thresholds" in config["validation"]
    assert "n_splits" in config["validation"]["cross_validation"]
    assert "shuffle" in config["validation"]["cross_validation"]
    assert "random_state" in config["validation"]["cross_validation"]
    assert "min_accuracy" in config["validation"]["thresholds"]
    assert "min_precision" in config["validation"]["thresholds"]
    assert "min_recall" in config["validation"]["thresholds"]
    assert "min_f1" in config["validation"]["thresholds"] 