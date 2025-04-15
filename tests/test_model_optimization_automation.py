import pytest
from src.model.model_optimization_automation import ModelOptimizationAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def optimization_automation():
    config_dir = "./config"
    optimization_dir = "./optimization"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(optimization_dir, exist_ok=True)
    
    config = {
        "default": {
            "hyperparameter_optimization": {
                "enabled": True,
                "method": "bayesian",
                "n_trials": 100,
                "metric": "accuracy",
                "search_space": {
                    "learning_rate": {
                        "type": "float",
                        "min": 0.001,
                        "max": 0.1
                    },
                    "n_estimators": {
                        "type": "int",
                        "min": 50,
                        "max": 500
                    },
                    "max_depth": {
                        "type": "int",
                        "min": 3,
                        "max": 10
                    }
                }
            },
            "feature_optimization": {
                "enabled": True,
                "method": "recursive",
                "n_features": 10,
                "scoring": "f1",
                "cv": 5
            },
            "model_architecture": {
                "enabled": True,
                "type": "ensemble",
                "models": {
                    "random_forest": {
                        "n_estimators": 100,
                        "max_depth": 10
                    },
                    "xgboost": {
                        "learning_rate": 0.1,
                        "n_estimators": 100
                    },
                    "lightgbm": {
                        "learning_rate": 0.1,
                        "n_estimators": 100
                    }
                },
                "voting": "soft"
            },
            "performance_optimization": {
                "enabled": True,
                "memory": {
                    "optimization": True,
                    "batch_size": 32
                },
                "parallel": {
                    "processing": True,
                    "n_jobs": -1
                }
            },
            "resource_optimization": {
                "enabled": True,
                "cpu": {
                    "cores": 4,
                    "threads": 8
                },
                "memory": {
                    "limit": "8GB"
                },
                "gpu": {
                    "enabled": False
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "save_path": "./optimization/logs"
            }
        }
    }
    with open(os.path.join(config_dir, "model_optimization_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelOptimizationAutomation(config_dir=config_dir, optimization_dir=optimization_dir)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]), y

@pytest.fixture
def sample_model():
    class MockModel:
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        
        def predict_proba(self, X):
            return np.random.rand(len(X), 2)
    
    return MockModel()

def test_optimization_initialization(optimization_automation):
    assert optimization_automation is not None
    assert optimization_automation.config_dir == "./config"
    assert optimization_automation.optimization_dir == "./optimization"

def test_hyperparameter_optimization_setup(optimization_automation):
    hp_config = optimization_automation.get_hyperparameter_optimization_config()
    assert hp_config is not None
    assert hp_config["enabled"] == True
    assert hp_config["method"] == "bayesian"
    assert hp_config["n_trials"] == 100

def test_feature_optimization_setup(optimization_automation):
    feature_config = optimization_automation.get_feature_optimization_config()
    assert feature_config is not None
    assert feature_config["enabled"] == True
    assert feature_config["method"] == "recursive"
    assert feature_config["n_features"] == 10

def test_model_architecture_setup(optimization_automation):
    arch_config = optimization_automation.get_model_architecture_config()
    assert arch_config is not None
    assert arch_config["enabled"] == True
    assert arch_config["type"] == "ensemble"
    assert arch_config["models"]["random_forest"]["n_estimators"] == 100

def test_performance_optimization_setup(optimization_automation):
    perf_config = optimization_automation.get_performance_optimization_config()
    assert perf_config is not None
    assert perf_config["enabled"] == True
    assert perf_config["memory"]["optimization"] == True
    assert perf_config["parallel"]["processing"] == True

def test_resource_optimization_setup(optimization_automation):
    resource_config = optimization_automation.get_resource_optimization_config()
    assert resource_config is not None
    assert resource_config["enabled"] == True
    assert resource_config["cpu"]["cores"] == 4
    assert resource_config["memory"]["limit"] == "8GB"

def test_logging_setup(optimization_automation):
    log_config = optimization_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./optimization/logs"

def test_hyperparameter_optimization_execution(optimization_automation, sample_data):
    X, y = sample_data
    results = optimization_automation.optimize_hyperparameters(X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "best_params" in results
    assert "best_score" in results

def test_feature_optimization_execution(optimization_automation, sample_data):
    X, y = sample_data
    results = optimization_automation.optimize_features(X, y)
    assert results is not None
    assert isinstance(results, tuple)
    assert len(results) == 2

def test_model_architecture_execution(optimization_automation, sample_data):
    X, y = sample_data
    results = optimization_automation.optimize_architecture(X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "model" in results
    assert "metrics" in results

def test_performance_optimization_execution(optimization_automation, sample_data, sample_model):
    X, y = sample_data
    results = optimization_automation.optimize_performance(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "memory_usage" in results
    assert "processing_time" in results

def test_resource_optimization_execution(optimization_automation, sample_data, sample_model):
    X, y = sample_data
    results = optimization_automation.optimize_resources(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "cpu_usage" in results
    assert "memory_usage" in results

def test_optimization_performance(optimization_automation, sample_data):
    X, y = sample_data
    start_time = datetime.now()
    optimization_automation.optimize_hyperparameters(X, y)
    optimization_automation.optimize_features(X, y)
    optimization_automation.optimize_architecture(X, y)
    optimization_automation.optimize_performance(sample_model, X, y)
    optimization_automation.optimize_resources(sample_model, X, y)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 3600.0

def test_error_handling(optimization_automation):
    with pytest.raises(ValueError):
        optimization_automation.set_hyperparameter_optimization_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        optimization_automation.set_feature_optimization_config({"invalid_config": {}})

def test_optimization_configuration(optimization_automation):
    config = optimization_automation.get_configuration()
    assert config is not None
    assert "hyperparameter_optimization" in config
    assert "feature_optimization" in config
    assert "model_architecture" in config
    assert "performance_optimization" in config
    assert "resource_optimization" in config
    assert "logging" in config 