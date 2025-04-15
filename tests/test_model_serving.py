import pytest
from src.model.model_serving import ModelServing
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_serving():
    config_dir = "./config"
    serving_dir = "./serving"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(serving_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "serving": {
                "api": {
                    "type": "rest",
                    "port": 8000,
                    "host": "localhost",
                    "endpoints": ["/predict", "/health", "/metrics"],
                    "timeout": 30
                },
                "scaling": {
                    "min_instances": 1,
                    "max_instances": 5,
                    "target_utilization": 0.7
                },
                "monitoring": {
                    "metrics": ["latency", "throughput", "error_rate"],
                    "interval": 60,
                    "alert_threshold": 0.9
                },
                "logging": {
                    "level": "info",
                    "format": "json",
                    "retention": 7
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_serving.json"), "w") as f:
        json.dump(config, f)
    
    return ModelServing(config_dir=config_dir, serving_dir=serving_dir)

@pytest.fixture
def sample_model():
    # 샘플 모델 클래스 생성
    class DummyModel:
        def predict(self, X):
            return np.random.rand(len(X))
    
    return DummyModel()

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    return X

def test_model_serving_initialization(model_serving):
    assert model_serving is not None
    assert model_serving.config_dir == "./config"
    assert model_serving.serving_dir == "./serving"

def test_api_setup(model_serving, sample_model):
    # API 설정 테스트
    api = model_serving.setup_api(sample_model)
    
    assert api is not None
    assert api.port == 8000
    assert api.host == "localhost"
    assert len(api.endpoints) == 3
    assert "/predict" in api.endpoints
    assert "/health" in api.endpoints
    assert "/metrics" in api.endpoints

def test_prediction_endpoint(model_serving, sample_model, sample_data):
    # 예측 엔드포인트 테스트
    api = model_serving.setup_api(sample_model)
    response = api.predict(sample_data)
    
    assert response is not None
    assert "predictions" in response
    assert len(response["predictions"]) == len(sample_data)

def test_health_endpoint(model_serving, sample_model):
    # 헬스 체크 엔드포인트 테스트
    api = model_serving.setup_api(sample_model)
    health = api.health_check()
    
    assert health is not None
    assert "status" in health
    assert health["status"] == "healthy"
    assert "timestamp" in health

def test_metrics_endpoint(model_serving, sample_model):
    # 메트릭 엔드포인트 테스트
    api = model_serving.setup_api(sample_model)
    metrics = api.get_metrics()
    
    assert metrics is not None
    assert "latency" in metrics
    assert "throughput" in metrics
    assert "error_rate" in metrics
    assert all(v >= 0 for v in metrics.values())

def test_scaling_setup(model_serving):
    # 스케일링 설정 테스트
    scaler = model_serving.setup_scaling()
    
    assert scaler is not None
    assert scaler.min_instances == 1
    assert scaler.max_instances == 5
    assert scaler.target_utilization == 0.7

def test_monitoring_setup(model_serving):
    # 모니터링 설정 테스트
    monitor = model_serving.setup_monitoring()
    
    assert monitor is not None
    assert "latency" in monitor.metrics
    assert "throughput" in monitor.metrics
    assert "error_rate" in monitor.metrics
    assert monitor.interval == 60
    assert monitor.alert_threshold == 0.9

def test_logging_setup(model_serving):
    # 로깅 설정 테스트
    logger = model_serving.setup_logging()
    
    assert logger is not None
    assert logger.level == "info"
    assert logger.format == "json"
    assert logger.retention == 7

def test_serving_performance(model_serving, sample_model, sample_data):
    # 서빙 성능 테스트
    start_time = datetime.now()
    
    # API 설정 및 예측 요청
    api = model_serving.setup_api(sample_model)
    for _ in range(100):
        api.predict(sample_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개 요청을 10초 이내에 처리
    assert processing_time < 10.0

def test_error_handling(model_serving):
    # 에러 처리 테스트
    # 잘못된 모델
    with pytest.raises(ValueError):
        model_serving.setup_api(None)
    
    # 잘못된 포트
    with pytest.raises(ValueError):
        model_serving.setup_api(sample_model, port=-1)
    
    # 잘못된 스케일링 설정
    with pytest.raises(ValueError):
        model_serving.setup_scaling(min_instances=-1)

def test_serving_configuration(model_serving):
    # 서빙 설정 테스트
    config = model_serving.get_configuration()
    
    assert config is not None
    assert "serving" in config
    assert "api" in config["serving"]
    assert "scaling" in config["serving"]
    assert "monitoring" in config["serving"]
    assert "logging" in config["serving"]
    assert "type" in config["serving"]["api"]
    assert "min_instances" in config["serving"]["scaling"]
    assert "metrics" in config["serving"]["monitoring"]
    assert "level" in config["serving"]["logging"] 