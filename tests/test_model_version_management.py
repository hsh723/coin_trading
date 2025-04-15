import pytest
from src.model.model_version_management import ModelVersionManagement
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def version_management():
    config_dir = "./config"
    model_dir = "./models"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    config = {
        "default": {
            "versioning": {
                "enabled": True,
                "format": "semantic",
                "auto_increment": True,
                "metadata": {
                    "author": "system",
                    "description": "auto-generated version"
                }
            },
            "storage": {
                "enabled": True,
                "type": "local",
                "path": "./models",
                "backup": {
                    "enabled": True,
                    "frequency": "daily",
                    "retention_days": 30
                }
            },
            "validation": {
                "enabled": True,
                "checks": ["format", "metadata", "dependencies"],
                "thresholds": {
                    "format": "strict",
                    "metadata": "required",
                    "dependencies": "exact"
                }
            },
            "rollback": {
                "enabled": True,
                "max_versions": 5,
                "auto_cleanup": True
            },
            "monitoring": {
                "enabled": True,
                "metrics": ["version_count", "storage_usage", "rollback_count"],
                "alert_thresholds": {
                    "version_count": 100,
                    "storage_usage": 1024,  # MB
                    "rollback_count": 3
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_version_management.json"), "w") as f:
        json.dump(config, f)
    
    return ModelVersionManagement(config_dir=config_dir, model_dir=model_dir)

@pytest.fixture
def sample_model():
    class MockModel:
        def __init__(self):
            self.version = "1.0.0"
            self.metadata = {
                "author": "test",
                "description": "test model",
                "created_at": datetime.now().isoformat()
            }
    
    return MockModel()

def test_version_management_initialization(version_management):
    assert version_management is not None
    assert version_management.config_dir == "./config"
    assert version_management.model_dir == "./models"

def test_versioning_setup(version_management):
    versioning_config = version_management.get_versioning_config()
    assert versioning_config is not None
    assert versioning_config["enabled"] == True
    assert versioning_config["format"] == "semantic"
    assert versioning_config["auto_increment"] == True

def test_storage_setup(version_management):
    storage_config = version_management.get_storage_config()
    assert storage_config is not None
    assert storage_config["enabled"] == True
    assert storage_config["type"] == "local"
    assert storage_config["backup"]["enabled"] == True

def test_validation_setup(version_management):
    validation_config = version_management.get_validation_config()
    assert validation_config is not None
    assert validation_config["enabled"] == True
    assert "format" in validation_config["checks"]
    assert validation_config["thresholds"]["format"] == "strict"

def test_rollback_setup(version_management):
    rollback_config = version_management.get_rollback_config()
    assert rollback_config is not None
    assert rollback_config["enabled"] == True
    assert rollback_config["max_versions"] == 5
    assert rollback_config["auto_cleanup"] == True

def test_monitoring_setup(version_management):
    monitoring_config = version_management.get_monitoring_config()
    assert monitoring_config is not None
    assert monitoring_config["enabled"] == True
    assert "version_count" in monitoring_config["metrics"]
    assert monitoring_config["alert_thresholds"]["version_count"] == 100

def test_version_creation(version_management, sample_model):
    version = version_management.create_version(sample_model)
    assert version is not None
    assert isinstance(version, str)
    assert version.startswith("1.0.0")

def test_version_retrieval(version_management, sample_model):
    version = version_management.create_version(sample_model)
    retrieved_model = version_management.get_version(version)
    assert retrieved_model is not None
    assert retrieved_model.version == version

def test_version_validation(version_management, sample_model):
    version = version_management.create_version(sample_model)
    validation_result = version_management.validate_version(version)
    assert validation_result is not None
    assert validation_result["status"] == "success"

def test_version_rollback(version_management, sample_model):
    version = version_management.create_version(sample_model)
    rollback_result = version_management.rollback_version(version)
    assert rollback_result is not None
    assert rollback_result["status"] == "success"

def test_version_cleanup(version_management, sample_model):
    version = version_management.create_version(sample_model)
    cleanup_result = version_management.cleanup_versions()
    assert cleanup_result is not None
    assert cleanup_result["status"] == "success"

def test_version_management_performance(version_management, sample_model):
    start_time = datetime.now()
    version_management.create_version(sample_model)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 5.0

def test_error_handling(version_management):
    with pytest.raises(ValueError):
        version_management.set_versioning_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        version_management.set_storage_config({"invalid_config": {}})

def test_version_management_configuration(version_management):
    config = version_management.get_configuration()
    assert config is not None
    assert "versioning" in config
    assert "storage" in config
    assert "validation" in config
    assert "rollback" in config
    assert "monitoring" in config 