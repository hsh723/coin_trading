import pytest
from src.model.model_version_manager import ModelVersionManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture
def version_manager():
    config_dir = "./config"
    versions_dir = "./versions"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(versions_dir, exist_ok=True)
    
    config = {
        "default": {
            "versioning": {
                "enabled": True,
                "format": "semantic",
                "auto_increment": {
                    "major": False,
                    "minor": True,
                    "patch": True
                }
            },
            "metadata": {
                "enabled": True,
                "required_fields": ["model_name", "framework", "input_shape", "output_shape"],
                "optional_fields": ["description", "author", "training_date", "performance_metrics"]
            },
            "storage": {
                "enabled": True,
                "local": {
                    "path": "./versions",
                    "format": "pickle"
                },
                "backup": {
                    "enabled": True,
                    "frequency": "daily",
                    "retention": 30
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_version_manager.json"), "w") as f:
        json.dump(config, f)
    
    return ModelVersionManager(config_dir=config_dir, versions_dir=versions_dir)

@pytest.fixture
def sample_model():
    class MockModel:
        def __init__(self):
            self.model_name = "test_model"
            self.framework = "tensorflow"
            self.input_shape = (10,)
            self.output_shape = (1,)
            self.description = "Test model for versioning"
            self.author = "Test Author"
            self.training_date = datetime.now()
            self.performance_metrics = {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93
            }
            self.hyperparameters = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            }
    
    return MockModel()

def test_manager_initialization(version_manager):
    assert version_manager is not None
    assert version_manager.config_dir == "./config"
    assert version_manager.versions_dir == "./versions"

def test_version_format_setup(version_manager):
    version_config = version_manager.get_version_format_config()
    assert version_config is not None
    assert version_config["enabled"] == True
    assert version_config["format"] == "semantic"
    assert version_config["auto_increment"]["minor"] == True

def test_metadata_setup(version_manager):
    metadata_config = version_manager.get_metadata_config()
    assert metadata_config is not None
    assert metadata_config["enabled"] == True
    assert "model_name" in metadata_config["required_fields"]
    assert "description" in metadata_config["optional_fields"]

def test_storage_setup(version_manager):
    storage_config = version_manager.get_storage_config()
    assert storage_config is not None
    assert storage_config["enabled"] == True
    assert storage_config["local"]["path"] == "./versions"
    assert storage_config["backup"]["enabled"] == True

def test_version_creation(version_manager, sample_model):
    version = version_manager.create_version(sample_model)
    assert version is not None
    assert isinstance(version, str)
    assert version.startswith("1.0.")

def test_metadata_storage(version_manager, sample_model):
    version = version_manager.create_version(sample_model)
    metadata = version_manager.get_metadata(version)
    assert metadata is not None
    assert metadata["model_name"] == sample_model.model_name
    assert metadata["framework"] == sample_model.framework

def test_version_listing(version_manager, sample_model):
    version_manager.create_version(sample_model)
    versions = version_manager.list_versions()
    assert versions is not None
    assert isinstance(versions, list)
    assert len(versions) > 0

def test_version_retrieval(version_manager, sample_model):
    version = version_manager.create_version(sample_model)
    retrieved_model = version_manager.get_version(version)
    assert retrieved_model is not None
    assert hasattr(retrieved_model, "model_name")
    assert hasattr(retrieved_model, "framework")

def test_version_comparison(version_manager, sample_model):
    version1 = version_manager.create_version(sample_model)
    sample_model.performance_metrics["accuracy"] = 0.96
    version2 = version_manager.create_version(sample_model)
    differences = version_manager.compare_versions(version1, version2)
    assert differences is not None
    assert "performance_metrics" in differences

def test_version_rollback(version_manager, sample_model):
    version1 = version_manager.create_version(sample_model)
    sample_model.performance_metrics["accuracy"] = 0.96
    version2 = version_manager.create_version(sample_model)
    rollback_result = version_manager.rollback_version(version1)
    assert rollback_result is not None
    assert rollback_result["success"] == True

def test_backup_creation(version_manager, sample_model):
    version = version_manager.create_version(sample_model)
    backup_result = version_manager.create_backup(version)
    assert backup_result is not None
    assert backup_result["success"] == True

def test_manager_performance(version_manager, sample_model):
    start_time = datetime.now()
    for _ in range(10):
        version_manager.create_version(sample_model)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 5.0

def test_error_handling(version_manager):
    with pytest.raises(ValueError):
        version_manager.set_version_format_config({"invalid_config": {}})
    with pytest.raises(ValueError):
        version_manager.set_metadata_config({"invalid_config": {}})

def test_manager_configuration(version_manager):
    config = version_manager.get_configuration()
    assert config is not None
    assert "versioning" in config
    assert "metadata" in config
    assert "storage" in config 