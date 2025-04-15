import pytest
from src.model.model_deployment_manager import ModelDeploymentManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def deployment_manager():
    config_dir = "./config"
    deploy_dir = "./deployments"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "deployment": {
                "environments": {
                    "development": {
                        "host": "localhost",
                        "port": 8000,
                        "replicas": 1
                    },
                    "staging": {
                        "host": "staging.example.com",
                        "port": 8000,
                        "replicas": 2
                    },
                    "production": {
                        "host": "api.example.com",
                        "port": 8000,
                        "replicas": 5
                    }
                },
                "rollout": {
                    "strategy": "canary",
                    "steps": [0.1, 0.25, 0.5, 0.75, 1.0],
                    "interval": 300
                },
                "rollback": {
                    "enabled": True,
                    "threshold": 0.9,
                    "timeout": 3600
                },
                "monitoring": {
                    "metrics": ["latency", "throughput", "error_rate"],
                    "interval": 60,
                    "alert_threshold": 0.9
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_deployment_manager.json"), "w") as f:
        json.dump(config, f)
    
    return ModelDeploymentManager(config_dir=config_dir, deploy_dir=deploy_dir)

@pytest.fixture
def sample_model():
    # 샘플 모델 클래스 생성
    class DummyModel:
        def __init__(self):
            self.name = "test_model"
            self.version = "1.0.0"
            self.metadata = {
                "author": "test",
                "description": "test model"
            }
    
    return DummyModel()

def test_deployment_manager_initialization(deployment_manager):
    assert deployment_manager is not None
    assert deployment_manager.config_dir == "./config"
    assert deployment_manager.deploy_dir == "./deployments"

def test_environment_setup(deployment_manager):
    # 환경 설정 테스트
    env = deployment_manager.setup_environment("development")
    
    assert env is not None
    assert env.host == "localhost"
    assert env.port == 8000
    assert env.replicas == 1

def test_deployment_creation(deployment_manager, sample_model):
    # 배포 생성 테스트
    deployment = deployment_manager.create_deployment(sample_model, "development")
    
    assert deployment is not None
    assert deployment.model.name == sample_model.name
    assert deployment.model.version == sample_model.version
    assert deployment.environment == "development"

def test_rollout_strategy(deployment_manager, sample_model):
    # 롤아웃 전략 테스트
    strategy = deployment_manager.setup_rollout_strategy()
    
    assert strategy is not None
    assert strategy.type == "canary"
    assert len(strategy.steps) == 5
    assert strategy.interval == 300

def test_rollback_strategy(deployment_manager):
    # 롤백 전략 테스트
    strategy = deployment_manager.setup_rollback_strategy()
    
    assert strategy is not None
    assert strategy.enabled is True
    assert strategy.threshold == 0.9
    assert strategy.timeout == 3600

def test_deployment_monitoring(deployment_manager):
    # 배포 모니터링 테스트
    monitor = deployment_manager.setup_monitoring()
    
    assert monitor is not None
    assert "latency" in monitor.metrics
    assert "throughput" in monitor.metrics
    assert "error_rate" in monitor.metrics
    assert monitor.interval == 60
    assert monitor.alert_threshold == 0.9

def test_deployment_status(deployment_manager, sample_model):
    # 배포 상태 테스트
    deployment = deployment_manager.create_deployment(sample_model, "development")
    status = deployment_manager.get_deployment_status(deployment.id)
    
    assert status is not None
    assert "state" in status
    assert "replicas" in status
    assert "metrics" in status

def test_deployment_rollback(deployment_manager, sample_model):
    # 배포 롤백 테스트
    deployment = deployment_manager.create_deployment(sample_model, "development")
    rollback = deployment_manager.rollback_deployment(deployment.id)
    
    assert rollback is not None
    assert rollback.success is True
    assert rollback.previous_version is not None

def test_deployment_performance(deployment_manager, sample_model):
    # 배포 성능 테스트
    start_time = datetime.now()
    
    # 배포 생성 및 상태 확인
    deployment = deployment_manager.create_deployment(sample_model, "development")
    deployment_manager.get_deployment_status(deployment.id)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 배포 생성 및 상태 확인을 5초 이내에 완료
    assert processing_time < 5.0

def test_error_handling(deployment_manager):
    # 에러 처리 테스트
    # 잘못된 환경
    with pytest.raises(ValueError):
        deployment_manager.setup_environment("invalid_env")
    
    # 잘못된 롤아웃 전략
    with pytest.raises(ValueError):
        deployment_manager.setup_rollout_strategy(strategy="invalid")
    
    # 잘못된 롤백 설정
    with pytest.raises(ValueError):
        deployment_manager.setup_rollback_strategy(threshold=-1)

def test_deployment_configuration(deployment_manager):
    # 배포 설정 테스트
    config = deployment_manager.get_configuration()
    
    assert config is not None
    assert "deployment" in config
    assert "environments" in config["deployment"]
    assert "rollout" in config["deployment"]
    assert "rollback" in config["deployment"]
    assert "monitoring" in config["deployment"]
    assert "development" in config["deployment"]["environments"]
    assert "strategy" in config["deployment"]["rollout"]
    assert "enabled" in config["deployment"]["rollback"]
    assert "metrics" in config["deployment"]["monitoring"] 