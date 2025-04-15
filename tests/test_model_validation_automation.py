import pytest
from src.model.model_validation_automation import ModelValidationAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def validation_automation():
    config_dir = "./config"
    validation_dir = "./validation"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    
    config = {
        "default": {
            "metrics": {
                "enabled": True,
                "classification": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.7,
                    "f1_score": 0.7,
                    "roc_auc": 0.8
                },
                "regression": {
                    "mse": 0.1,
                    "rmse": 0.3,
                    "mae": 0.2,
                    "r2": 0.8,
                    "explained_variance": 0.8
                }
            },
            "cross_validation": {
                "enabled": True,
                "method": "kfold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42
            },
            "stability": {
                "enabled": True,
                "bootstrap": {
                    "n_samples": 100,
                    "sample_size": 0.8
                },
                "perturbation": {
                    "noise_level": 0.1,
                    "n_samples": 100
                }
            },
            "bias": {
                "enabled": True,
                "metrics": {
                    "demographic_parity": 0.1,
                    "equal_opportunity": 0.1,
                    "predictive_equality": 0.1
                },
                "protected_attributes": {
                    "age": {"bins": [18, 25, 35, 45, 55, 65]},
                    "gender": ["male", "female", "other"],
                    "race": ["white", "black", "asian", "hispanic"]
                }
            },
            "interpretability": {
                "enabled": True,
                "feature_importance": {
                    "method": "permutation",
                    "n_repeats": 10,
                    "random_state": 42
                },
                "shap_values": {
                    "n_samples": 100,
                    "background_samples": 50
                },
                "partial_dependence": {
                    "features": "all",
                    "grid_resolution": 20
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "save_path": "./validation/logs"
            }
        }
    }
    with open(os.path.join(config_dir, "model_validation_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelValidationAutomation(config_dir=config_dir, validation_dir=validation_dir)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]), y

@pytest.fixture
def sample_model():
    class MockModel:
        def __init__(self):
            self.coef_ = np.random.randn(10)
            self.intercept_ = np.random.randn()
        
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        
        def predict_proba(self, X):
            return np.random.rand(len(X), 2)
    
    return MockModel()

def test_validation_initialization(validation_automation):
    assert validation_automation is not None
    assert validation_automation.config_dir == "./config"
    assert validation_automation.validation_dir == "./validation"

def test_metrics_setup(validation_automation):
    metrics_config = validation_automation.get_metrics_config()
    assert metrics_config is not None
    assert metrics_config["enabled"] == True
    assert metrics_config["classification"]["accuracy"] == 0.8
    assert metrics_config["regression"]["r2"] == 0.8

def test_cross_validation_setup(validation_automation):
    cv_config = validation_automation.get_cross_validation_config()
    assert cv_config is not None
    assert cv_config["enabled"] == True
    assert cv_config["method"] == "kfold"
    assert cv_config["n_splits"] == 5

def test_stability_setup(validation_automation):
    stability_config = validation_automation.get_stability_config()
    assert stability_config is not None
    assert stability_config["enabled"] == True
    assert stability_config["bootstrap"]["n_samples"] == 100
    assert stability_config["perturbation"]["noise_level"] == 0.1

def test_bias_setup(validation_automation):
    bias_config = validation_automation.get_bias_config()
    assert bias_config is not None
    assert bias_config["enabled"] == True
    assert bias_config["metrics"]["demographic_parity"] == 0.1
    assert "age" in bias_config["protected_attributes"]

def test_interpretability_setup(validation_automation):
    interp_config = validation_automation.get_interpretability_config()
    assert interp_config is not None
    assert interp_config["enabled"] == True
    assert interp_config["feature_importance"]["method"] == "permutation"
    assert interp_config["shap_values"]["n_samples"] == 100

def test_logging_setup(validation_automation):
    log_config = validation_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./validation/logs"

def test_metrics_evaluation(validation_automation, sample_model, sample_data):
    X, y = sample_data
    metrics_result = validation_automation.evaluate_metrics(sample_model, X, y)
    assert metrics_result is not None
    assert isinstance(metrics_result, dict)
    assert "classification" in metrics_result
    assert "regression" in metrics_result

def test_cross_validation_execution(validation_automation, sample_model, sample_data):
    X, y = sample_data
    cv_result = validation_automation.execute_cross_validation(sample_model, X, y)
    assert cv_result is not None
    assert isinstance(cv_result, dict)
    assert "scores" in cv_result
    assert "mean_score" in cv_result

def test_stability_evaluation(validation_automation, sample_model, sample_data):
    X, y = sample_data
    stability_result = validation_automation.evaluate_stability(sample_model, X, y)
    assert stability_result is not None
    assert isinstance(stability_result, dict)
    assert "bootstrap" in stability_result
    assert "perturbation" in stability_result

def test_bias_evaluation(validation_automation, sample_model, sample_data):
    X, y = sample_data
    bias_result = validation_automation.evaluate_bias(sample_model, X, y)
    assert bias_result is not None
    assert isinstance(bias_result, dict)
    assert "demographic_parity" in bias_result
    assert "equal_opportunity" in bias_result

def test_interpretability_evaluation(validation_automation, sample_model, sample_data):
    X, y = sample_data
    interp_result = validation_automation.evaluate_interpretability(sample_model, X, y)
    assert interp_result is not None
    assert isinstance(interp_result, dict)
    assert "feature_importance" in interp_result
    assert "shap_values" in interp_result

def test_validation_performance(validation_automation, sample_model, sample_data):
    X, y = sample_data
    start_time = datetime.now()
    validation_automation.evaluate_metrics(sample_model, X, y)
    validation_automation.execute_cross_validation(sample_model, X, y)
    validation_automation.evaluate_stability(sample_model, X, y)
    validation_automation.evaluate_bias(sample_model, X, y)
    validation_automation.evaluate_interpretability(sample_model, X, y)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 3600.0

def test_error_handling(validation_automation):
    with pytest.raises(ValueError):
        validation_automation.set_metrics_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        validation_automation.set_cross_validation_config({"invalid_config": {}})

def test_validation_configuration(validation_automation):
    config = validation_automation.get_configuration()
    assert config is not None
    assert "metrics" in config
    assert "cross_validation" in config
    assert "stability" in config
    assert "bias" in config
    assert "interpretability" in config
    assert "logging" in config 