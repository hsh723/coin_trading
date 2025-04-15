import pytest
from src.model.model_training_automation import ModelTrainingAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def training_automation():
    config_dir = "./config"
    training_dir = "./training"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    
    config = {
        "default": {
            "data_preparation": {
                "enabled": True,
                "steps": [
                    "data_loading",
                    "data_cleaning",
                    "feature_engineering",
                    "data_splitting"
                ],
                "parameters": {
                    "test_size": 0.2,
                    "validation_size": 0.1,
                    "random_state": 42
                }
            },
            "model_selection": {
                "enabled": True,
                "algorithms": [
                    "random_forest",
                    "xgboost",
                    "lightgbm",
                    "logistic_regression"
                ],
                "selection_criteria": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score"
                ]
            },
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "grid_search",
                "parameters": {
                    "n_iter": 10,
                    "cv": 5,
                    "scoring": "accuracy"
                }
            },
            "training": {
                "enabled": True,
                "parameters": {
                    "batch_size": 32,
                    "epochs": 100,
                    "early_stopping": {
                        "enabled": True,
                        "patience": 10,
                        "min_delta": 0.001
                    }
                }
            },
            "evaluation": {
                "enabled": True,
                "metrics": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ],
                "thresholds": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.7,
                    "f1_score": 0.7,
                    "roc_auc": 0.8
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "save_path": "./training/logs"
            }
        }
    }
    with open(os.path.join(config_dir, "model_training_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelTrainingAutomation(config_dir=config_dir, training_dir=training_dir)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]), y

def test_training_initialization(training_automation):
    assert training_automation is not None
    assert training_automation.config_dir == "./config"
    assert training_automation.training_dir == "./training"

def test_data_preparation_setup(training_automation):
    prep_config = training_automation.get_data_preparation_config()
    assert prep_config is not None
    assert prep_config["enabled"] == True
    assert "data_loading" in prep_config["steps"]
    assert prep_config["parameters"]["test_size"] == 0.2

def test_model_selection_setup(training_automation):
    selection_config = training_automation.get_model_selection_config()
    assert selection_config is not None
    assert selection_config["enabled"] == True
    assert "random_forest" in selection_config["algorithms"]
    assert "accuracy" in selection_config["selection_criteria"]

def test_hyperparameter_tuning_setup(training_automation):
    tuning_config = training_automation.get_hyperparameter_tuning_config()
    assert tuning_config is not None
    assert tuning_config["enabled"] == True
    assert tuning_config["method"] == "grid_search"
    assert tuning_config["parameters"]["n_iter"] == 10

def test_training_setup(training_automation):
    training_config = training_automation.get_training_config()
    assert training_config is not None
    assert training_config["enabled"] == True
    assert training_config["parameters"]["batch_size"] == 32
    assert training_config["parameters"]["early_stopping"]["enabled"] == True

def test_evaluation_setup(training_automation):
    eval_config = training_automation.get_evaluation_config()
    assert eval_config is not None
    assert eval_config["enabled"] == True
    assert "accuracy" in eval_config["metrics"]
    assert eval_config["thresholds"]["accuracy"] == 0.8

def test_logging_setup(training_automation):
    log_config = training_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./training/logs"

def test_data_preparation_execution(training_automation, sample_data):
    X, y = sample_data
    prepared_data = training_automation.execute_data_preparation(X, y)
    assert prepared_data is not None
    assert isinstance(prepared_data, dict)
    assert "X_train" in prepared_data
    assert "X_test" in prepared_data
    assert "y_train" in prepared_data
    assert "y_test" in prepared_data

def test_model_selection_execution(training_automation, sample_data):
    X, y = sample_data
    selected_model = training_automation.execute_model_selection(X, y)
    assert selected_model is not None
    assert isinstance(selected_model, dict)
    assert "algorithm" in selected_model
    assert "parameters" in selected_model

def test_hyperparameter_tuning_execution(training_automation, sample_data):
    X, y = sample_data
    best_params = training_automation.execute_hyperparameter_tuning(X, y)
    assert best_params is not None
    assert isinstance(best_params, dict)

def test_training_execution(training_automation, sample_data):
    X, y = sample_data
    model = training_automation.execute_training(X, y)
    assert model is not None
    assert hasattr(model, "predict")

def test_evaluation_execution(training_automation, sample_data):
    X, y = sample_data
    metrics = training_automation.execute_evaluation(X, y)
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics

def test_training_performance(training_automation, sample_data):
    X, y = sample_data
    start_time = datetime.now()
    training_automation.execute_data_preparation(X, y)
    training_automation.execute_model_selection(X, y)
    training_automation.execute_hyperparameter_tuning(X, y)
    training_automation.execute_training(X, y)
    training_automation.execute_evaluation(X, y)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 300.0

def test_error_handling(training_automation):
    with pytest.raises(ValueError):
        training_automation.set_data_preparation_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        training_automation.set_model_selection_config({"invalid_config": {}})

def test_training_configuration(training_automation):
    config = training_automation.get_configuration()
    assert config is not None
    assert "data_preparation" in config
    assert "model_selection" in config
    assert "hyperparameter_tuning" in config
    assert "training" in config
    assert "evaluation" in config
    assert "logging" in config 