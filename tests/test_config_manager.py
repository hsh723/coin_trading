import pytest
from src.utils.config_manager import ConfigManager
import os
import json
import yaml
import time

@pytest.fixture
def config_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성 (JSON)
    json_config = {
        "default": {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "trading": {
                "max_position_size": 1.0,
                "max_leverage": 20,
                "risk_limit": 0.02
            }
        }
    }
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(json_config, f)
    
    # YAML 설정 파일 생성
    yaml_config = {
        "default": {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "test_user",
                "password": "test_password"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    }
    with open(os.path.join(config_dir, "config.yaml"), "w") as f:
        yaml.dump(yaml_config, f)
    
    return ConfigManager(config_dir=config_dir, data_dir=data_dir)

def test_config_manager_initialization(config_manager):
    assert config_manager is not None
    assert config_manager.config_dir == "./config"
    assert config_manager.data_dir == "./data"

def test_config_loading(config_manager):
    # JSON 설정 로드 테스트
    json_config = config_manager.load_config("config.json")
    assert json_config is not None
    assert "default" in json_config
    assert "api_key" in json_config["default"]
    assert "trading" in json_config["default"]
    
    # YAML 설정 로드 테스트
    yaml_config = config_manager.load_config("config.yaml")
    assert yaml_config is not None
    assert "default" in yaml_config
    assert "database" in yaml_config["default"]
    assert "logging" in yaml_config["default"]

def test_config_saving(config_manager):
    # 새로운 설정 저장 테스트
    new_config = {
        "test": {
            "key1": "value1",
            "key2": "value2"
        }
    }
    
    config_manager.save_config("test_config.json", new_config)
    loaded_config = config_manager.load_config("test_config.json")
    assert loaded_config == new_config

def test_config_updating(config_manager):
    # 설정 업데이트 테스트
    updates = {
        "default": {
            "trading": {
                "max_position_size": 2.0
            }
        }
    }
    
    config_manager.update_config("config.json", updates)
    updated_config = config_manager.load_config("config.json")
    assert updated_config["default"]["trading"]["max_position_size"] == 2.0

def test_config_validation(config_manager):
    # 설정 유효성 검사 테스트
    valid_config = {
        "default": {
            "api_key": "valid_key",
            "api_secret": "valid_secret"
        }
    }
    assert config_manager.validate_config(valid_config) is True
    
    invalid_config = {
        "default": {
            "api_key": None,
            "api_secret": ""
        }
    }
    assert config_manager.validate_config(invalid_config) is False

def test_config_encryption(config_manager):
    # 설정 암호화 테스트
    sensitive_data = {
        "api_key": "sensitive_key",
        "api_secret": "sensitive_secret"
    }
    
    encrypted_data = config_manager.encrypt_config(sensitive_data)
    assert encrypted_data != sensitive_data
    
    decrypted_data = config_manager.decrypt_config(encrypted_data)
    assert decrypted_data == sensitive_data

def test_config_backup(config_manager):
    # 설정 백업 테스트
    config_manager.backup_config("config.json")
    
    backup_dir = os.path.join(config_manager.data_dir, "backups")
    assert os.path.exists(backup_dir)
    
    backup_files = os.listdir(backup_dir)
    assert len(backup_files) > 0
    assert any("config.json" in f for f in backup_files)

def test_config_restore(config_manager):
    # 설정 복원 테스트
    # 먼저 백업 생성
    config_manager.backup_config("config.json")
    
    # 설정 수정
    original_config = config_manager.load_config("config.json")
    modified_config = original_config.copy()
    modified_config["default"]["api_key"] = "modified_key"
    config_manager.save_config("config.json", modified_config)
    
    # 최신 백업으로 복원
    config_manager.restore_config("config.json")
    restored_config = config_manager.load_config("config.json")
    assert restored_config == original_config

def test_config_watching(config_manager):
    # 설정 변경 감지 테스트
    config_file = "config.json"
    original_config = config_manager.load_config(config_file)
    
    # 설정 변경 감지 시작
    config_manager.start_watching(config_file)
    
    # 설정 수정
    modified_config = original_config.copy()
    modified_config["default"]["api_key"] = "new_key"
    config_manager.save_config(config_file, modified_config)
    
    # 잠시 대기하여 변경 감지
    time.sleep(1)
    
    # 변경 감지 확인
    assert config_manager.has_config_changed(config_file) is True
    
    # 감지 중지
    config_manager.stop_watching(config_file)

def test_config_merging(config_manager):
    # 설정 병합 테스트
    base_config = {
        "default": {
            "key1": "value1",
            "key2": "value2"
        }
    }
    
    override_config = {
        "default": {
            "key2": "new_value2",
            "key3": "value3"
        }
    }
    
    merged_config = config_manager.merge_configs(base_config, override_config)
    assert merged_config["default"]["key1"] == "value1"
    assert merged_config["default"]["key2"] == "new_value2"
    assert merged_config["default"]["key3"] == "value3" 