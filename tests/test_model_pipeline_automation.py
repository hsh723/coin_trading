import pytest
from src.model.model_pipeline_automation import ModelPipelineAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def pipeline_automation():
    config_dir = "./config"
    pipeline_dir = "./pipeline"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(pipeline_dir, exist_ok=True)
    
    config = {
        "default": {
            "data_preprocessing": {
                "enabled": True,
                "steps": {
                    "missing_values": {
                        "strategy": "mean",
                        "threshold": 0.1
                    },
                    "outliers": {
                        "method": "zscore",
                        "threshold": 3.0
                    },
                    "scaling": {
                        "method": "standard",
                        "features": "all"
                    }
                }
            },
            "feature_selection": {
                "enabled": True,
                "method": "recursive",
                "n_features": 10,
                "scoring": "f1",
                "cv": 5
            },
            "model_training": {
                "enabled": True,
                "models": {
                    "classification": {
                        "RandomForest": {
                            "n_estimators": 100,
                            "max_depth": 10
                        },
                        "XGBoost": {
                            "n_estimators": 100,
                            "max_depth": 6
                        }
                    },
                    "regression": {
                        "LinearRegression": {},
                        "RandomForestRegressor": {
                            "n_estimators": 100,
                            "max_depth": 10
                        }
                    }
                }
            },
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "bayesian",
                "n_iter": 50,
                "cv": 5,
                "scoring": "f1"
            },
            "model_evaluation": {
                "enabled": True,
                "metrics": {
                    "classification": {
                        "accuracy": {},
                        "precision": {},
                        "recall": {},
                        "f1": {}
                    },
                    "regression": {
                        "mse": {},
                        "mae": {},
                        "r2": {}
                    }
                },
                "cross_validation": {
                    "n_splits": 5,
                    "shuffle": True
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "save_path": "./pipeline/logs"
            }
        }
    }
    with open(os.path.join(config_dir, "model_pipeline_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelPipelineAutomation(config_dir=config_dir, pipeline_dir=pipeline_dir)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]), y

def test_pipeline_initialization(pipeline_automation):
    assert pipeline_automation is not None
    assert pipeline_automation.config_dir == "./config"
    assert pipeline_automation.pipeline_dir == "./pipeline"

def test_data_preprocessing_setup(pipeline_automation):
    preprocessing_config = pipeline_automation.get_preprocessing_config()
    assert preprocessing_config is not None
    assert preprocessing_config["enabled"] == True
    assert preprocessing_config["steps"]["missing_values"]["strategy"] == "mean"
    assert preprocessing_config["steps"]["outliers"]["method"] == "zscore"

def test_feature_selection_setup(pipeline_automation):
    feature_config = pipeline_automation.get_feature_selection_config()
    assert feature_config is not None
    assert feature_config["enabled"] == True
    assert feature_config["method"] == "recursive"
    assert feature_config["n_features"] == 10

def test_model_training_setup(pipeline_automation):
    training_config = pipeline_automation.get_model_training_config()
    assert training_config is not None
    assert training_config["enabled"] == True
    assert "RandomForest" in training_config["models"]["classification"]
    assert "LinearRegression" in training_config["models"]["regression"]

def test_hyperparameter_tuning_setup(pipeline_automation):
    tuning_config = pipeline_automation.get_hyperparameter_tuning_config()
    assert tuning_config is not None
    assert tuning_config["enabled"] == True
    assert tuning_config["method"] == "bayesian"
    assert tuning_config["n_iter"] == 50

def test_model_evaluation_setup(pipeline_automation):
    evaluation_config = pipeline_automation.get_model_evaluation_config()
    assert evaluation_config is not None
    assert evaluation_config["enabled"] == True
    assert "accuracy" in evaluation_config["metrics"]["classification"]
    assert "mse" in evaluation_config["metrics"]["regression"]

def test_logging_setup(pipeline_automation):
    log_config = pipeline_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./pipeline/logs"

def test_data_preprocessing_execution(pipeline_automation, sample_data):
    X, y = sample_data
    results = pipeline_automation.preprocess_data(X)
    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert not results.isnull().any().any()

def test_feature_selection_execution(pipeline_automation, sample_data):
    X, y = sample_data
    results = pipeline_automation.select_features(X, y)
    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert results.shape[1] <= X.shape[1]

def test_model_training_execution(pipeline_automation, sample_data):
    X, y = sample_data
    results = pipeline_automation.train_models(X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "models" in results
    assert "metrics" in results

def test_hyperparameter_tuning_execution(pipeline_automation, sample_data):
    X, y = sample_data
    results = pipeline_automation.tune_hyperparameters(X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "best_params" in results
    assert "best_score" in results

def test_model_evaluation_execution(pipeline_automation, sample_data):
    X, y = sample_data
    results = pipeline_automation.evaluate_models(X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "metrics" in results
    assert "cv_scores" in results

def test_pipeline_performance(pipeline_automation, sample_data):
    X, y = sample_data
    start_time = datetime.now()
    pipeline_automation.run_pipeline(X, y)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 3600.0

def test_error_handling(pipeline_automation):
    with pytest.raises(ValueError):
        pipeline_automation.set_preprocessing_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        pipeline_automation.set_feature_selection_config({"invalid_config": {}})

def test_pipeline_configuration(pipeline_automation):
    config = pipeline_automation.get_configuration()
    assert config is not None
    assert "data_preprocessing" in config
    assert "feature_selection" in config
    assert "model_training" in config
    assert "hyperparameter_tuning" in config
    assert "model_evaluation" in config
    assert "logging" in config 