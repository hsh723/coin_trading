import pytest
from src.model.model_deployer import ModelDeployer
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_deployer():
    config_dir = "./config"
    model_dir = "./models"
    deploy_dir = "./deploy"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "deployment": {
                "format": "onnx",
                "target": "cpu",
                "optimization": {
                    "quantization": True,
                    "pruning": False,
                    "compression": True
                },
                "api": {
                    "type": "rest",
                    "port": 8000,
                    "host": "localhost",
                    "endpoints": ["/predict", "/health"]
                },
                "monitoring": {
                    "metrics": ["latency", "throughput", "memory"],
                    "interval": 60,
                    "alert_threshold": 0.9
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_deployer.json"), "w") as f:
        json.dump(config, f)
    
    return ModelDeployer(config_dir=config_dir, model_dir=model_dir, deploy_dir=deploy_dir)

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

def test_model_deployer_initialization(model_deployer):
    assert model_deployer is not None
    assert model_deployer.config_dir == "./config"
    assert model_deployer.model_dir == "./models"
    assert model_deployer.deploy_dir == "./deploy"

def test_model_conversion(model_deployer, sample_model):
    # 모델 변환 테스트
    converted_model = model_deployer.convert_model(sample_model)
    
    assert converted_model is not None
    assert os.path.exists(os.path.join(model_deployer.deploy_dir, "model.onnx"))

def test_model_optimization(model_deployer, sample_model):
    # 모델 최적화 테스트
    optimized_model = model_deployer.optimize_model(sample_model)
    
    assert optimized_model is not None
    assert os.path.exists(os.path.join(model_deployer.deploy_dir, "optimized_model.onnx"))

def test_api_generation(model_deployer, sample_model):
    # API 생성 테스트
    api = model_deployer.generate_api(sample_model)
    
    assert api is not None
    assert os.path.exists(os.path.join(model_deployer.deploy_dir, "api.py"))
    assert os.path.exists(os.path.join(model_deployer.deploy_dir, "requirements.txt"))

def test_api_endpoints(model_deployer, sample_model):
    # API 엔드포인트 테스트
    api = model_deployer.generate_api(sample_model)
    
    assert "/predict" in api.endpoints
    assert "/health" in api.endpoints
    
    # 예측 엔드포인트 테스트
    response = api.predict(sample_data)
    assert response is not None
    assert "prediction" in response
    
    # 헬스 체크 엔드포인트 테스트
    health = api.health_check()
    assert health is not None
    assert "status" in health
    assert health["status"] == "healthy"

def test_model_serving(model_deployer, sample_model):
    # 모델 서빙 테스트
    server = model_deployer.serve_model(sample_model)
    
    assert server is not None
    assert server.is_running()
    assert server.port == 8000
    assert server.host == "localhost"

def test_monitoring_setup(model_deployer):
    # 모니터링 설정 테스트
    monitor = model_deployer.setup_monitoring()
    
    assert monitor is not None
    assert "latency" in monitor.metrics
    assert "throughput" in monitor.metrics
    assert "memory" in monitor.metrics
    assert monitor.interval == 60
    assert monitor.alert_threshold == 0.9

def test_deployment_performance(model_deployer, sample_model, sample_data):
    # 배포 성능 테스트
    start_time = datetime.now()
    
    # 모델 변환 및 서빙
    converted_model = model_deployer.convert_model(sample_model)
    server = model_deployer.serve_model(converted_model)
    
    # 예측 요청
    for _ in range(100):
        server.predict(sample_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 100개 요청을 10초 이내에 처리
    assert processing_time < 10.0

def test_error_handling(model_deployer):
    # 에러 처리 테스트
    # 잘못된 모델
    with pytest.raises(ValueError):
        model_deployer.convert_model(None)
    
    # 잘못된 포트
    with pytest.raises(ValueError):
        model_deployer.serve_model(sample_model, port=-1)
    
    # 잘못된 최적화 설정
    with pytest.raises(ValueError):
        model_deployer.optimize_model(sample_model, optimization={"invalid": True})

def test_deployment_configuration(model_deployer):
    # 배포 설정 테스트
    config = model_deployer.get_configuration()
    
    assert config is not None
    assert "deployment" in config
    assert "format" in config["deployment"]
    assert "target" in config["deployment"]
    assert "optimization" in config["deployment"]
    assert "api" in config["deployment"]
    assert "monitoring" in config["deployment"]
    assert "quantization" in config["deployment"]["optimization"]
    assert "type" in config["deployment"]["api"]
    assert "metrics" in config["deployment"]["monitoring"] 