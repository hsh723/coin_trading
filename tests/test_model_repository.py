import pytest
from src.model.model_repository import ModelRepository
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_repository():
    config_dir = "./config"
    repo_dir = "./repository"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(repo_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "repository": {
                "storage": {
                    "type": "local",
                    "path": "./repository",
                    "backup": {
                        "enabled": True,
                        "interval": 24,
                        "retention": 30
                    }
                },
                "metadata": {
                    "required": ["name", "version", "author", "description"],
                    "optional": ["tags", "notes"]
                },
                "access": {
                    "permissions": ["read", "write", "delete"],
                    "users": ["admin", "developer"]
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_repository.json"), "w") as f:
        json.dump(config, f)
    
    return ModelRepository(config_dir=config_dir, repo_dir=repo_dir)

@pytest.fixture
def sample_model():
    # 샘플 모델 클래스 생성
    class DummyModel:
        def __init__(self):
            self.name = "test_model"
            self.version = "1.0.0"
            self.metadata = {
                "author": "test",
                "description": "test model",
                "tags": ["regression", "test"]
            }
    
    return DummyModel()

def test_model_repository_initialization(model_repository):
    assert model_repository is not None
    assert model_repository.config_dir == "./config"
    assert model_repository.repo_dir == "./repository"

def test_model_storage(model_repository, sample_model):
    # 모델 저장 테스트
    storage_path = model_repository.store_model(sample_model)
    
    assert storage_path is not None
    assert os.path.exists(storage_path)
    assert os.path.exists(os.path.join(storage_path, "model.pkl"))
    assert os.path.exists(os.path.join(storage_path, "metadata.json"))

def test_model_retrieval(model_repository, sample_model):
    # 모델 검색 테스트
    storage_path = model_repository.store_model(sample_model)
    retrieved_model = model_repository.get_model(sample_model.name, sample_model.version)
    
    assert retrieved_model is not None
    assert retrieved_model.name == sample_model.name
    assert retrieved_model.version == sample_model.version
    assert retrieved_model.metadata["author"] == sample_model.metadata["author"]

def test_model_listing(model_repository, sample_model):
    # 모델 목록 테스트
    # 여러 모델 저장
    models = []
    for i in range(3):
        model = sample_model
        model.version = f"1.0.{i}"
        model_repository.store_model(model)
        models.append((model.name, model.version))
    
    # 모델 목록 조회
    model_list = model_repository.list_models()
    
    assert model_list is not None
    assert isinstance(model_list, list)
    assert len(model_list) >= 3
    assert all((m["name"], m["version"]) in models for m in model_list)

def test_model_metadata(model_repository, sample_model):
    # 모델 메타데이터 테스트
    storage_path = model_repository.store_model(sample_model)
    metadata = model_repository.get_model_metadata(sample_model.name, sample_model.version)
    
    assert metadata is not None
    assert "name" in metadata
    assert "version" in metadata
    assert "author" in metadata
    assert "description" in metadata
    assert metadata["name"] == sample_model.name
    assert metadata["version"] == sample_model.version
    assert metadata["author"] == sample_model.metadata["author"]

def test_model_backup(model_repository, sample_model):
    # 모델 백업 테스트
    storage_path = model_repository.store_model(sample_model)
    backup_path = model_repository.backup_model(sample_model.name, sample_model.version)
    
    assert backup_path is not None
    assert os.path.exists(backup_path)
    assert os.path.exists(os.path.join(backup_path, "model.pkl"))
    assert os.path.exists(os.path.join(backup_path, "metadata.json"))

def test_model_deletion(model_repository, sample_model):
    # 모델 삭제 테스트
    storage_path = model_repository.store_model(sample_model)
    model_repository.delete_model(sample_model.name, sample_model.version)
    
    assert not os.path.exists(storage_path)
    assert not os.path.exists(os.path.join(storage_path, "model.pkl"))
    assert not os.path.exists(os.path.join(storage_path, "metadata.json"))

def test_repository_performance(model_repository, sample_model):
    # 저장소 성능 테스트
    start_time = datetime.now()
    
    # 모델 저장 및 검색
    storage_path = model_repository.store_model(sample_model)
    model_repository.get_model(sample_model.name, sample_model.version)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 모델 저장 및 검색을 2초 이내에 완료
    assert processing_time < 2.0

def test_error_handling(model_repository):
    # 에러 처리 테스트
    # 잘못된 모델
    with pytest.raises(ValueError):
        model_repository.store_model(None)
    
    # 존재하지 않는 모델
    with pytest.raises(ValueError):
        model_repository.get_model("invalid_model", "1.0.0")
    
    # 잘못된 메타데이터
    with pytest.raises(ValueError):
        model_repository.store_model(sample_model, metadata={"invalid": True})

def test_repository_configuration(model_repository):
    # 저장소 설정 테스트
    config = model_repository.get_configuration()
    
    assert config is not None
    assert "repository" in config
    assert "storage" in config["repository"]
    assert "metadata" in config["repository"]
    assert "access" in config["repository"]
    assert "type" in config["repository"]["storage"]
    assert "required" in config["repository"]["metadata"]
    assert "permissions" in config["repository"]["access"] 