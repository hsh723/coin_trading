import pytest
from src.model.model_versioner import ModelVersioner
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_versioner():
    config_dir = "./config"
    model_dir = "./models"
    version_dir = "./versions"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(version_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "versioning": {
                "format": "semantic",
                "storage": {
                    "type": "local",
                    "path": "./versions"
                },
                "metadata": {
                    "required": ["author", "description", "performance"],
                    "optional": ["tags", "notes"]
                },
                "backup": {
                    "enabled": True,
                    "interval": 24,
                    "retention": 30
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_versioner.json"), "w") as f:
        json.dump(config, f)
    
    return ModelVersioner(config_dir=config_dir, model_dir=model_dir, version_dir=version_dir)

@pytest.fixture
def sample_model():
    # 샘플 모델 클래스 생성
    class DummyModel:
        def __init__(self):
            self.version = "1.0.0"
            self.metadata = {
                "author": "test",
                "description": "test model",
                "performance": {"accuracy": 0.9}
            }
    
    return DummyModel()

def test_model_versioner_initialization(model_versioner):
    assert model_versioner is not None
    assert model_versioner.config_dir == "./config"
    assert model_versioner.model_dir == "./models"
    assert model_versioner.version_dir == "./versions"

def test_version_creation(model_versioner, sample_model):
    # 버전 생성 테스트
    version = model_versioner.create_version(sample_model)
    
    assert version is not None
    assert os.path.exists(os.path.join(model_versioner.version_dir, version))
    assert os.path.exists(os.path.join(model_versioner.version_dir, version, "model.pkl"))
    assert os.path.exists(os.path.join(model_versioner.version_dir, version, "metadata.json"))

def test_version_metadata(model_versioner, sample_model):
    # 버전 메타데이터 테스트
    version = model_versioner.create_version(sample_model)
    metadata = model_versioner.get_version_metadata(version)
    
    assert metadata is not None
    assert "author" in metadata
    assert "description" in metadata
    assert "performance" in metadata
    assert metadata["author"] == "test"
    assert metadata["description"] == "test model"
    assert metadata["performance"]["accuracy"] == 0.9

def test_version_listing(model_versioner, sample_model):
    # 버전 목록 테스트
    # 여러 버전 생성
    versions = []
    for i in range(3):
        version = model_versioner.create_version(sample_model)
        versions.append(version)
    
    # 버전 목록 조회
    version_list = model_versioner.list_versions()
    
    assert version_list is not None
    assert isinstance(version_list, list)
    assert len(version_list) >= 3
    assert all(v in version_list for v in versions)

def test_version_retrieval(model_versioner, sample_model):
    # 버전 검색 테스트
    version = model_versioner.create_version(sample_model)
    retrieved_model = model_versioner.get_version(version)
    
    assert retrieved_model is not None
    assert retrieved_model.version == version
    assert retrieved_model.metadata["author"] == "test"

def test_version_comparison(model_versioner, sample_model):
    # 버전 비교 테스트
    version1 = model_versioner.create_version(sample_model)
    version2 = model_versioner.create_version(sample_model)
    
    comparison = model_versioner.compare_versions(version1, version2)
    
    assert comparison is not None
    assert "differences" in comparison
    assert "metadata" in comparison
    assert "performance" in comparison

def test_version_rollback(model_versioner, sample_model):
    # 버전 롤백 테스트
    version1 = model_versioner.create_version(sample_model)
    version2 = model_versioner.create_version(sample_model)
    
    # 버전2로 롤백
    model_versioner.rollback_version(version2)
    
    # 현재 버전 확인
    current_version = model_versioner.get_current_version()
    assert current_version == version2

def test_version_backup(model_versioner, sample_model):
    # 버전 백업 테스트
    version = model_versioner.create_version(sample_model)
    backup_path = model_versioner.backup_version(version)
    
    assert backup_path is not None
    assert os.path.exists(backup_path)
    assert os.path.exists(os.path.join(backup_path, "model.pkl"))
    assert os.path.exists(os.path.join(backup_path, "metadata.json"))

def test_versioning_performance(model_versioner, sample_model):
    # 버전 관리 성능 테스트
    start_time = datetime.now()
    
    # 버전 생성 및 검색
    version = model_versioner.create_version(sample_model)
    model_versioner.get_version(version)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 버전 생성 및 검색을 2초 이내에 완료
    assert processing_time < 2.0

def test_error_handling(model_versioner):
    # 에러 처리 테스트
    # 잘못된 버전 형식
    with pytest.raises(ValueError):
        model_versioner.create_version(None)
    
    # 존재하지 않는 버전
    with pytest.raises(ValueError):
        model_versioner.get_version("invalid_version")
    
    # 잘못된 메타데이터
    with pytest.raises(ValueError):
        model_versioner.create_version(sample_model, metadata={"invalid": True})

def test_versioning_configuration(model_versioner):
    # 버전 관리 설정 테스트
    config = model_versioner.get_configuration()
    
    assert config is not None
    assert "versioning" in config
    assert "format" in config["versioning"]
    assert "storage" in config["versioning"]
    assert "metadata" in config["versioning"]
    assert "backup" in config["versioning"]
    assert "type" in config["versioning"]["storage"]
    assert "required" in config["versioning"]["metadata"]
    assert "enabled" in config["versioning"]["backup"] 