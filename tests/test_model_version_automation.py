import pytest
from src.model.model_version_automation import ModelVersionAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def version_automation():
    config_dir = "./config"
    version_dir = "./versions"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(version_dir, exist_ok=True)
    
    config = {
        "default": {
            "versioning": {
                "enabled": True,
                "format": "semantic",
                "parameters": {
                    "major": 1,
                    "minor": 0,
                    "patch": 0
                },
                "max_versions": 10,
                "retention_days": 30
            },
            "metadata": {
                "enabled": True,
                "required_fields": [
                    "model_type",
                    "framework",
                    "training_date",
                    "performance_metrics",
                    "hyperparameters",
                    "data_schema"
                ]
            },
            "storage": {
                "enabled": True,
                "type": "local",
                "path": "./versions",
                "compression": {
                    "enabled": True,
                    "format": "zip",
                    "level": 9
                }
            },
            "validation": {
                "enabled": True,
                "checks": [
                    "performance_threshold",
                    "data_compatibility",
                    "dependency_check"
                ],
                "thresholds": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.7,
                    "f1_score": 0.7
                }
            },
            "backup": {
                "enabled": True,
                "frequency": "daily",
                "retention": {
                    "days": 7,
                    "max_backups": 5
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "save_path": "./versions/logs"
            }
        }
    }
    with open(os.path.join(config_dir, "model_version_automation.json"), "w") as f:
        json.dump(config, f)
    
    return ModelVersionAutomation(config_dir=config_dir, version_dir=version_dir)

@pytest.fixture
def sample_model():
    class MockModel:
        def __init__(self):
            self.coef_ = np.random.randn(10)
            self.intercept_ = np.random.randn()
            self.metadata = {
                "model_type": "classification",
                "framework": "scikit-learn",
                "training_date": datetime.now().isoformat(),
                "performance_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.83,
                    "f1_score": 0.82
                },
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1
                },
                "data_schema": {
                    "features": ["feature_1", "feature_2", "feature_3"],
                    "target": "label"
                }
            }
    
    return MockModel()

def test_version_initialization(version_automation):
    assert version_automation is not None
    assert version_automation.config_dir == "./config"
    assert version_automation.version_dir == "./versions"

def test_versioning_setup(version_automation):
    version_config = version_automation.get_versioning_config()
    assert version_config is not None
    assert version_config["enabled"] == True
    assert version_config["format"] == "semantic"
    assert version_config["parameters"]["major"] == 1
    assert version_config["max_versions"] == 10

def test_metadata_setup(version_automation):
    metadata_config = version_automation.get_metadata_config()
    assert metadata_config is not None
    assert metadata_config["enabled"] == True
    assert "model_type" in metadata_config["required_fields"]
    assert "framework" in metadata_config["required_fields"]

def test_storage_setup(version_automation):
    storage_config = version_automation.get_storage_config()
    assert storage_config is not None
    assert storage_config["enabled"] == True
    assert storage_config["type"] == "local"
    assert storage_config["compression"]["enabled"] == True

def test_validation_setup(version_automation):
    validation_config = version_automation.get_validation_config()
    assert validation_config is not None
    assert validation_config["enabled"] == True
    assert "performance_threshold" in validation_config["checks"]
    assert validation_config["thresholds"]["accuracy"] == 0.8

def test_backup_setup(version_automation):
    backup_config = version_automation.get_backup_config()
    assert backup_config is not None
    assert backup_config["enabled"] == True
    assert backup_config["frequency"] == "daily"
    assert backup_config["retention"]["days"] == 7

def test_logging_setup(version_automation):
    log_config = version_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./versions/logs"

def test_version_creation(version_automation, sample_model):
    version_result = version_automation.create_version(sample_model)
    assert version_result is not None
    assert isinstance(version_result, dict)
    assert "version" in version_result
    assert "status" in version_result

def test_version_retrieval(version_automation, sample_model):
    version_result = version_automation.create_version(sample_model)
    retrieved_version = version_automation.get_version(version_result["version"])
    assert retrieved_version is not None
    assert isinstance(retrieved_version, dict)
    assert "model" in retrieved_version
    assert "metadata" in retrieved_version

def test_version_validation(version_automation, sample_model):
    validation_result = version_automation.validate_version(sample_model)
    assert validation_result is not None
    assert isinstance(validation_result, dict)
    assert "status" in validation_result
    assert "checks" in validation_result

def test_version_backup(version_automation, sample_model):
    backup_result = version_automation.backup_version(sample_model)
    assert backup_result is not None
    assert isinstance(backup_result, dict)
    assert "status" in backup_result
    assert "backup_path" in backup_result

def test_version_performance(version_automation, sample_model):
    start_time = datetime.now()
    version_automation.create_version(sample_model)
    version_automation.validate_version(sample_model)
    version_automation.backup_version(sample_model)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 300.0

def test_error_handling(version_automation):
    with pytest.raises(ValueError):
        version_automation.set_versioning_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        version_automation.set_metadata_config({"invalid_config": {}})

def test_version_configuration(version_automation):
    config = version_automation.get_configuration()
    assert config is not None
    assert "versioning" in config
    assert "metadata" in config
    assert "storage" in config
    assert "validation" in config
    assert "backup" in config
    assert "logging" in config 