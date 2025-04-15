import pytest
from src.model.model_monitoring_automation import ModelMonitoringAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def monitoring_automation():
    config_dir = "./config"
    monitoring_dir = "./monitoring"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(monitoring_dir, exist_ok=True)
    
    config = {
        "default": {
            "performance_metrics": {
                "enabled": True,
                "metrics": {
                    "accuracy": {
                        "threshold": 0.8,
                        "check_interval": "1h"
                    },
                    "precision": {
                        "threshold": 0.7,
                        "check_interval": "1h"
                    },
                    "recall": {
                        "threshold": 0.7,
                        "check_interval": "1h"
                    },
                    "f1": {
                        "threshold": 0.7,
                        "check_interval": "1h"
                    }
                }
            },
            "drift_metrics": {
                "enabled": True,
                "methods": {
                    "statistical": {
                        "ks_test": {
                            "threshold": 0.05,
                            "check_interval": "1d"
                        },
                        "chi_squared": {
                            "threshold": 0.05,
                            "check_interval": "1d"
                        }
                    },
                    "distance": {
                        "wasserstein": {
                            "threshold": 0.1,
                            "check_interval": "1d"
                        },
                        "kl_divergence": {
                            "threshold": 0.1,
                            "check_interval": "1d"
                        }
                    }
                }
            },
            "data_quality": {
                "enabled": True,
                "checks": {
                    "missing_values": {
                        "threshold": 0.1,
                        "check_interval": "1h"
                    },
                    "outliers": {
                        "method": "zscore",
                        "threshold": 3.0,
                        "check_interval": "1h"
                    },
                    "data_types": {
                        "check_interval": "1h"
                    }
                }
            },
            "alerts": {
                "enabled": True,
                "channels": {
                    "email": {
                        "recipients": ["admin@example.com"],
                        "thresholds": {
                            "performance": 0.1,
                            "drift": 0.1,
                            "data_quality": 0.1
                        }
                    },
                    "slack": {
                        "webhook_url": "https://hooks.slack.com/services/xxx",
                        "channel": "#monitoring",
                        "thresholds": {
                            "performance": 0.1,
                            "drift": 0.1,
                            "data_quality": 0.1
                        }
                    }
                }
            },
            "visualization": {
                "enabled": True,
                "metrics": {
                    "performance": {
                        "period": "7d",
                        "save_path": "./monitoring/plots/performance"
                    },
                    "drift": {
                        "period": "30d",
                        "save_path": "./monitoring/plots/drift"
                    },
                    "data_quality": {
                        "period": "7d",
                        "save_path": "./monitoring/plots/data_quality"
                    }
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "save_path": "./monitoring/logs"
            }
        }
    }
    with open(os.path.join(config_dir, "model_monitoring_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelMonitoringAutomation(config_dir=config_dir, monitoring_dir=monitoring_dir)

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

def test_monitoring_initialization(monitoring_automation):
    assert monitoring_automation is not None
    assert monitoring_automation.config_dir == "./config"
    assert monitoring_automation.monitoring_dir == "./monitoring"

def test_performance_metrics_setup(monitoring_automation):
    metrics_config = monitoring_automation.get_performance_metrics_config()
    assert metrics_config is not None
    assert metrics_config["enabled"] == True
    assert metrics_config["metrics"]["accuracy"]["threshold"] == 0.8
    assert metrics_config["metrics"]["accuracy"]["check_interval"] == "1h"

def test_drift_metrics_setup(monitoring_automation):
    drift_config = monitoring_automation.get_drift_metrics_config()
    assert drift_config is not None
    assert drift_config["enabled"] == True
    assert drift_config["methods"]["statistical"]["ks_test"]["threshold"] == 0.05
    assert drift_config["methods"]["distance"]["wasserstein"]["threshold"] == 0.1

def test_data_quality_setup(monitoring_automation):
    quality_config = monitoring_automation.get_data_quality_config()
    assert quality_config is not None
    assert quality_config["enabled"] == True
    assert quality_config["checks"]["missing_values"]["threshold"] == 0.1
    assert quality_config["checks"]["outliers"]["method"] == "zscore"

def test_alerts_setup(monitoring_automation):
    alerts_config = monitoring_automation.get_alerts_config()
    assert alerts_config is not None
    assert alerts_config["enabled"] == True
    assert alerts_config["channels"]["email"]["recipients"] == ["admin@example.com"]
    assert alerts_config["channels"]["slack"]["channel"] == "#monitoring"

def test_visualization_setup(monitoring_automation):
    viz_config = monitoring_automation.get_visualization_config()
    assert viz_config is not None
    assert viz_config["enabled"] == True
    assert viz_config["metrics"]["performance"]["period"] == "7d"
    assert viz_config["metrics"]["drift"]["period"] == "30d"

def test_logging_setup(monitoring_automation):
    log_config = monitoring_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./monitoring/logs"

def test_performance_monitoring_execution(monitoring_automation, sample_data, sample_model):
    X, y = sample_data
    results = monitoring_automation.monitor_performance(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "metrics" in results
    assert "alerts" in results

def test_drift_monitoring_execution(monitoring_automation, sample_data):
    X, _ = sample_data
    results = monitoring_automation.monitor_drift(X)
    assert results is not None
    assert isinstance(results, dict)
    assert "statistical" in results
    assert "distance" in results

def test_data_quality_monitoring_execution(monitoring_automation, sample_data):
    X, _ = sample_data
    results = monitoring_automation.monitor_data_quality(X)
    assert results is not None
    assert isinstance(results, dict)
    assert "missing_values" in results
    assert "outliers" in results

def test_alert_sending_execution(monitoring_automation):
    results = monitoring_automation.send_alerts("test_alert", "Test message")
    assert results is not None
    assert isinstance(results, dict)
    assert "email" in results
    assert "slack" in results

def test_visualization_generation_execution(monitoring_automation):
    results = monitoring_automation.generate_visualizations()
    assert results is not None
    assert isinstance(results, dict)
    assert "performance" in results
    assert "drift" in results
    assert "data_quality" in results

def test_monitoring_performance(monitoring_automation, sample_data, sample_model):
    X, y = sample_data
    start_time = datetime.now()
    monitoring_automation.run_monitoring(sample_model, X, y)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 3600.0

def test_error_handling(monitoring_automation):
    with pytest.raises(ValueError):
        monitoring_automation.set_performance_metrics_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        monitoring_automation.set_drift_metrics_config({"invalid_config": {}})

def test_monitoring_configuration(monitoring_automation):
    config = monitoring_automation.get_configuration()
    assert config is not None
    assert "performance_metrics" in config
    assert "drift_metrics" in config
    assert "data_quality" in config
    assert "alerts" in config
    assert "visualization" in config
    assert "logging" in config 