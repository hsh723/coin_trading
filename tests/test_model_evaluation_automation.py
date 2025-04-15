import pytest
from src.model.model_evaluation_automation import ModelEvaluationAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def evaluation_automation():
    config_dir = "./config"
    evaluation_dir = "./evaluation"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    
    config = {
        "default": {
            "metrics": {
                "enabled": True,
                "classification": {
                    "accuracy": True,
                    "precision": True,
                    "recall": True,
                    "f1": True,
                    "roc_auc": True
                },
                "regression": {
                    "mse": True,
                    "rmse": True,
                    "mae": True,
                    "r2": True,
                    "explained_variance": True
                }
            },
            "cross_validation": {
                "enabled": True,
                "method": "kfold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42
            },
            "bias_fairness": {
                "enabled": True,
                "metrics": {
                    "demographic_parity": True,
                    "equal_opportunity": True,
                    "predictive_equality": True
                },
                "protected_attributes": {
                    "age": True,
                    "gender": True,
                    "race": True
                }
            },
            "robustness": {
                "enabled": True,
                "adversarial": {
                    "method": "fgsm",
                    "epsilon": 0.1
                },
                "perturbation": {
                    "noise": {
                        "type": "gaussian",
                        "std": 0.1
                    }
                },
                "outliers": {
                    "method": "isolation_forest",
                    "contamination": 0.1
                }
            },
            "interpretability": {
                "enabled": True,
                "feature_importance": {
                    "method": "permutation",
                    "n_repeats": 10
                },
                "shap": {
                    "enabled": True,
                    "background_samples": 100
                }
            },
            "comparison": {
                "enabled": True,
                "baseline_models": {
                    "dummy": True,
                    "random": True
                },
                "statistical_tests": {
                    "t_test": True,
                    "wilcoxon": True
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "save_path": "./evaluation/logs"
            }
        }
    }
    with open(os.path.join(config_dir, "model_evaluation_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelEvaluationAutomation(config_dir=config_dir, evaluation_dir=evaluation_dir)

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
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        
        def predict_proba(self, X):
            return np.random.rand(len(X), 2)
    
    return MockModel()

def test_evaluation_initialization(evaluation_automation):
    assert evaluation_automation is not None
    assert evaluation_automation.config_dir == "./config"
    assert evaluation_automation.evaluation_dir == "./evaluation"

def test_metrics_setup(evaluation_automation):
    metrics_config = evaluation_automation.get_metrics_config()
    assert metrics_config is not None
    assert metrics_config["enabled"] == True
    assert metrics_config["classification"]["accuracy"] == True
    assert metrics_config["regression"]["mse"] == True

def test_cross_validation_setup(evaluation_automation):
    cv_config = evaluation_automation.get_cross_validation_config()
    assert cv_config is not None
    assert cv_config["enabled"] == True
    assert cv_config["method"] == "kfold"
    assert cv_config["n_splits"] == 5

def test_bias_fairness_setup(evaluation_automation):
    bias_config = evaluation_automation.get_bias_fairness_config()
    assert bias_config is not None
    assert bias_config["enabled"] == True
    assert bias_config["metrics"]["demographic_parity"] == True
    assert bias_config["protected_attributes"]["age"] == True

def test_robustness_setup(evaluation_automation):
    robust_config = evaluation_automation.get_robustness_config()
    assert robust_config is not None
    assert robust_config["enabled"] == True
    assert robust_config["adversarial"]["method"] == "fgsm"
    assert robust_config["perturbation"]["noise"]["type"] == "gaussian"

def test_interpretability_setup(evaluation_automation):
    interp_config = evaluation_automation.get_interpretability_config()
    assert interp_config is not None
    assert interp_config["enabled"] == True
    assert interp_config["feature_importance"]["method"] == "permutation"
    assert interp_config["shap"]["enabled"] == True

def test_comparison_setup(evaluation_automation):
    comp_config = evaluation_automation.get_comparison_config()
    assert comp_config is not None
    assert comp_config["enabled"] == True
    assert comp_config["baseline_models"]["dummy"] == True
    assert comp_config["statistical_tests"]["t_test"] == True

def test_logging_setup(evaluation_automation):
    log_config = evaluation_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./evaluation/logs"

def test_metrics_evaluation_execution(evaluation_automation, sample_data, sample_model):
    X, y = sample_data
    results = evaluation_automation.evaluate_metrics(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "classification" in results
    assert "regression" in results

def test_cross_validation_execution(evaluation_automation, sample_data, sample_model):
    X, y = sample_data
    results = evaluation_automation.evaluate_cross_validation(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "scores" in results
    assert "mean_score" in results

def test_bias_fairness_execution(evaluation_automation, sample_data, sample_model):
    X, y = sample_data
    results = evaluation_automation.evaluate_bias_fairness(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "demographic_parity" in results
    assert "equal_opportunity" in results

def test_robustness_execution(evaluation_automation, sample_data, sample_model):
    X, y = sample_data
    results = evaluation_automation.evaluate_robustness(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "adversarial" in results
    assert "perturbation" in results

def test_interpretability_execution(evaluation_automation, sample_data, sample_model):
    X, y = sample_data
    results = evaluation_automation.evaluate_interpretability(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "feature_importance" in results
    assert "shap_values" in results

def test_comparison_execution(evaluation_automation, sample_data, sample_model):
    X, y = sample_data
    results = evaluation_automation.evaluate_comparison(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "baseline_comparison" in results
    assert "statistical_tests" in results

def test_evaluation_performance(evaluation_automation, sample_data, sample_model):
    X, y = sample_data
    start_time = datetime.now()
    evaluation_automation.evaluate_metrics(sample_model, X, y)
    evaluation_automation.evaluate_cross_validation(sample_model, X, y)
    evaluation_automation.evaluate_bias_fairness(sample_model, X, y)
    evaluation_automation.evaluate_robustness(sample_model, X, y)
    evaluation_automation.evaluate_interpretability(sample_model, X, y)
    evaluation_automation.evaluate_comparison(sample_model, X, y)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 3600.0

def test_error_handling(evaluation_automation):
    with pytest.raises(ValueError):
        evaluation_automation.set_metrics_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        evaluation_automation.set_cross_validation_config({"invalid_config": {}})

def test_evaluation_configuration(evaluation_automation):
    config = evaluation_automation.get_configuration()
    assert config is not None
    assert "metrics" in config
    assert "cross_validation" in config
    assert "bias_fairness" in config
    assert "robustness" in config
    assert "interpretability" in config
    assert "comparison" in config
    assert "logging" in config 