import pytest
from src.model.model_performance_analyzer import ModelPerformanceAnalyzer
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def performance_analyzer():
    config_dir = "./config"
    analysis_dir = "./analysis"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    config = {
        "default": {
            "classification_metrics": {
                "enabled": True,
                "metrics": ["accuracy", "precision", "recall", "f1_score", "auc"],
                "thresholds": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.7,
                    "f1_score": 0.7,
                    "auc": 0.7
                },
                "average": "weighted"
            },
            "regression_metrics": {
                "enabled": True,
                "metrics": ["mse", "rmse", "mae", "r2"],
                "thresholds": {
                    "mse": 0.1,
                    "rmse": 0.3,
                    "mae": 0.2,
                    "r2": 0.7
                }
            },
            "clustering_metrics": {
                "enabled": True,
                "metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"],
                "thresholds": {
                    "silhouette": 0.5,
                    "calinski_harabasz": 100,
                    "davies_bouldin": 1.0
                }
            },
            "feature_importance": {
                "enabled": True,
                "methods": ["permutation", "shap"],
                "top_features": 5,
                "threshold": 0.1
            },
            "visualization": {
                "enabled": True,
                "types": ["confusion_matrix", "roc_curve", "feature_importance"],
                "save_format": "png",
                "dpi": 300
            }
        }
    }
    with open(os.path.join(config_dir, "model_performance_analyzer.json"), "w") as f:
        json.dump(config, f)
    
    return ModelPerformanceAnalyzer(config_dir=config_dir, analysis_dir=analysis_dir)

@pytest.fixture
def classification_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]), y

@pytest.fixture
def regression_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]), y

@pytest.fixture
def clustering_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])

def test_analyzer_initialization(performance_analyzer):
    assert performance_analyzer is not None
    assert performance_analyzer.config_dir == "./config"
    assert performance_analyzer.analysis_dir == "./analysis"

def test_classification_metrics_setup(performance_analyzer):
    metrics_config = performance_analyzer.get_classification_metrics_config()
    assert metrics_config is not None
    assert metrics_config["enabled"] == True
    assert "accuracy" in metrics_config["metrics"]
    assert metrics_config["thresholds"]["accuracy"] == 0.8

def test_regression_metrics_setup(performance_analyzer):
    metrics_config = performance_analyzer.get_regression_metrics_config()
    assert metrics_config is not None
    assert metrics_config["enabled"] == True
    assert "mse" in metrics_config["metrics"]
    assert metrics_config["thresholds"]["mse"] == 0.1

def test_clustering_metrics_setup(performance_analyzer):
    metrics_config = performance_analyzer.get_clustering_metrics_config()
    assert metrics_config is not None
    assert metrics_config["enabled"] == True
    assert "silhouette" in metrics_config["metrics"]
    assert metrics_config["thresholds"]["silhouette"] == 0.5

def test_feature_importance_setup(performance_analyzer):
    importance_config = performance_analyzer.get_feature_importance_config()
    assert importance_config is not None
    assert importance_config["enabled"] == True
    assert "permutation" in importance_config["methods"]
    assert importance_config["top_features"] == 5

def test_visualization_setup(performance_analyzer):
    viz_config = performance_analyzer.get_visualization_config()
    assert viz_config is not None
    assert viz_config["enabled"] == True
    assert "confusion_matrix" in viz_config["types"]
    assert viz_config["dpi"] == 300

def test_classification_metrics_calculation(performance_analyzer, classification_data):
    X, y = classification_data
    metrics = performance_analyzer.calculate_classification_metrics(X, y)
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics

def test_regression_metrics_calculation(performance_analyzer, regression_data):
    X, y = regression_data
    metrics = performance_analyzer.calculate_regression_metrics(X, y)
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "mse" in metrics

def test_clustering_metrics_calculation(performance_analyzer, clustering_data):
    X = clustering_data
    metrics = performance_analyzer.calculate_clustering_metrics(X)
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "silhouette" in metrics

def test_feature_importance_calculation(performance_analyzer, classification_data):
    X, y = classification_data
    importance = performance_analyzer.calculate_feature_importance(X, y)
    assert importance is not None
    assert isinstance(importance, dict)
    assert len(importance) <= 5

def test_visualization_generation(performance_analyzer, classification_data):
    X, y = classification_data
    visualizations = performance_analyzer.generate_visualizations(X, y)
    assert visualizations is not None
    assert isinstance(visualizations, dict)
    assert "confusion_matrix" in visualizations

def test_performance_report_generation(performance_analyzer, classification_data):
    X, y = classification_data
    report = performance_analyzer.generate_performance_report(X, y)
    assert report is not None
    assert isinstance(report, dict)
    assert "metrics" in report
    assert "visualizations" in report

def test_analyzer_performance(performance_analyzer, classification_data):
    X, y = classification_data
    start_time = datetime.now()
    performance_analyzer.calculate_classification_metrics(X, y)
    performance_analyzer.calculate_feature_importance(X, y)
    performance_analyzer.generate_visualizations(X, y)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 10.0

def test_error_handling(performance_analyzer):
    with pytest.raises(ValueError):
        performance_analyzer.set_classification_metrics_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        performance_analyzer.set_regression_metrics_config({"invalid_config": {}})

def test_analyzer_configuration(performance_analyzer):
    config = performance_analyzer.get_configuration()
    assert config is not None
    assert "classification_metrics" in config
    assert "regression_metrics" in config
    assert "clustering_metrics" in config
    assert "feature_importance" in config
    assert "visualization" in config 