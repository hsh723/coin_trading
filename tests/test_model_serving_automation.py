import pytest
from src.model.model_serving_automation import ModelServingAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def serving_automation():
    config_dir = "./config"
    serving_dir = "./serving"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(serving_dir, exist_ok=True)
    
    config = {
        "default": {
            "api": {
                "enabled": True,
                "endpoints": {
                    "predict": {
                        "path": "/predict",
                        "methods": ["POST"],
                        "rate_limit": 100
                    },
                    "health": {
                        "path": "/health",
                        "methods": ["GET"],
                        "rate_limit": 1000
                    },
                    "metrics": {
                        "path": "/metrics",
                        "methods": ["GET"],
                        "rate_limit": 1000
                    }
                },
                "authentication": {
                    "enabled": True,
                    "method": "jwt",
                    "token_expiry": 3600
                }
            },
            "scaling": {
                "enabled": True,
                "strategy": "auto",
                "min_instances": 1,
                "max_instances": 5,
                "metrics": ["cpu_usage", "memory_usage", "request_latency"],
                "thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 80,
                    "request_latency": 1000
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": ["latency", "throughput", "error_rate"],
                "alert_thresholds": {
                    "latency": 1000,
                    "throughput": 100,
                    "error_rate": 0.05
                },
                "logging": {
                    "level": "info",
                    "format": "json",
                    "retention_days": 30
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_serving_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelServingAutomation(config_dir=config_dir, serving_dir=serving_dir)

@pytest.fixture
def sample_model():
    class MockModel:
        def __init__(self):
            self.version = "1.0.0"
        
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
    
    return MockModel()

@pytest.fixture
def sample_request():
    np.random.seed(42)
    n_samples = 10
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])

def test_serving_initialization(serving_automation):
    assert serving_automation is not None
    assert serving_automation.config_dir == "./config"
    assert serving_automation.serving_dir == "./serving"

def test_api_setup(serving_automation):
    api_config = serving_automation.get_api_config()
    assert api_config is not None
    assert api_config["enabled"] == True
    assert "predict" in api_config["endpoints"]
    assert api_config["authentication"]["enabled"] == True

def test_scaling_setup(serving_automation):
    scaling_config = serving_automation.get_scaling_config()
    assert scaling_config is not None
    assert scaling_config["enabled"] == True
    assert scaling_config["strategy"] == "auto"
    assert scaling_config["max_instances"] == 5

def test_monitoring_setup(serving_automation):
    monitoring_config = serving_automation.get_monitoring_config()
    assert monitoring_config is not None
    assert monitoring_config["enabled"] == True
    assert "latency" in monitoring_config["metrics"]
    assert monitoring_config["logging"]["level"] == "info"

def test_model_serving(serving_automation, sample_model):
    serving_result = serving_automation.serve_model(sample_model)
    assert serving_result is not None
    assert isinstance(serving_result, dict)
    assert "status" in serving_result
    assert "endpoints" in serving_result

def test_prediction_endpoint(serving_automation, sample_model, sample_request):
    serving_automation.serve_model(sample_model)
    prediction = serving_automation.predict(sample_request)
    assert prediction is not None
    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == len(sample_request)

def test_health_check(serving_automation, sample_model):
    serving_automation.serve_model(sample_model)
    health_status = serving_automation.check_health()
    assert health_status is not None
    assert isinstance(health_status, dict)
    assert "status" in health_status
    assert health_status["status"] == "healthy"

def test_metrics_endpoint(serving_automation, sample_model):
    serving_automation.serve_model(sample_model)
    metrics = serving_automation.get_metrics()
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "latency" in metrics
    assert "throughput" in metrics
    assert "error_rate" in metrics

def test_serving_performance(serving_automation, sample_model, sample_request):
    serving_automation.serve_model(sample_model)
    start_time = datetime.now()
    for _ in range(10):
        serving_automation.predict(sample_request)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 5.0

def test_error_handling(serving_automation):
    with pytest.raises(ValueError):
        serving_automation.set_api_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        serving_automation.set_scaling_config({"invalid_config": {}})

def test_serving_configuration(serving_automation):
    config = serving_automation.get_configuration()
    assert config is not None
    assert "api" in config
    assert "scaling" in config
    assert "monitoring" in config 