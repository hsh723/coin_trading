import pytest
from src.model.model_deployment_automation import ModelDeploymentAutomation
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
import docker

@pytest.fixture
def deployment_automation():
    config_dir = "./config"
    deployment_dir = "c:/coin_Trading"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(deployment_dir, exist_ok=True)
    
    config = {
        "deployment_path": deployment_dir,
        "environment": {
            "enabled": True,
            "settings": {
                "python_version": "3.8",
                "dependencies": {
                    "numpy": "1.21.0",
                    "pandas": "1.3.0",
                    "scikit-learn": "0.24.2"
                },
                "requirements_file": "requirements.txt"
            }
        },
        "docker": {
            "enabled": True,
            "image": "python:3.8-slim",
            "ports": ["8080:8080"],
            "volumes": ["./models:/app/models"],
            "environment": {
                "MODEL_PATH": "/app/models",
                "LOG_LEVEL": "INFO"
            },
            "health_check": {
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            }
        },
        "security": {
            "enabled": True,
            "ssl": {
                "enabled": True,
                "cert_path": "/etc/ssl/certs",
                "key_path": "/etc/ssl/private"
            },
            "firewall": {
                "enabled": True,
                "rules": [
                    {
                        "port": 8080,
                        "protocol": "tcp",
                        "action": "allow"
                    }
                ]
            }
        },
        "monitoring": {
            "enabled": True,
            "metrics": {
                "prometheus": {
                    "enabled": True,
                    "port": 9090,
                    "path": "/metrics"
                },
                "grafana": {
                    "enabled": True,
                    "port": 3000,
                    "dashboard": "model_deployment"
                }
            }
        },
        "backup": {
            "enabled": True,
            "schedule": "0 0 * * *",
            "retention": 7,
            "storage": {
                "type": "s3",
                "bucket": "model-backups",
                "path": "daily"
            },
            "encryption": {
                "enabled": True,
                "algorithm": "AES-256"
            }
        },
        "logging": {
            "enabled": True,
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "save_path": "./deployment/logs",
            "rotation": {
                "max_size": "100MB",
                "backup_count": 5
            }
        }
    }
    
    config_path = os.path.join(config_dir, "model_deployment_automation.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    return ModelDeploymentAutomation(config_path=config_path)

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

def test_deployment_initialization(deployment_automation):
    assert deployment_automation is not None
    assert deployment_automation.config_dir == "./config"
    assert deployment_automation.deployment_dir == "c:/coin_Trading"

def test_environment_setup():
    """환경 설정 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 환경 설정 실행
    result = automation.setup_environment()
    
    # 결과 검증
    assert result is True
    
    # 환경 설정 조회
    config = automation.get_environment_config()
    
    # 설정 검증
    assert config["enabled"] is True
    assert config["settings"]["python_version"] == "3.8"
    assert config["settings"]["dependencies"] == {
        "numpy": "1.21.0",
        "pandas": "1.3.0",
        "scikit-learn": "0.24.2"
    }
    assert config["settings"]["docker"]["enabled"] is True
    assert config["settings"]["docker"]["image"] == "python:3.8-slim"
    assert config["settings"]["docker"]["health_check"]["interval"] == "30s"
    assert config["settings"]["docker"]["scaling"]["enabled"] is True
    assert config["settings"]["docker"]["networking"]["enabled"] is True
    assert config["settings"]["docker"]["security"]["enabled"] is True
    assert config["settings"]["docker"]["logging"]["enabled"] is True
    assert config["settings"]["docker"]["metrics"]["enabled"] is True
    assert config["settings"]["docker"]["backup"]["enabled"] is True
    assert config["settings"]["docker"]["disaster_recovery"]["enabled"] is True

def test_validation_setup(deployment_automation):
    validation_config = deployment_automation.get_validation_config()
    assert validation_config is not None
    assert validation_config["enabled"] == True
    assert validation_config["tests"]["accuracy"]["threshold"] == 0.8
    assert validation_config["tests"]["latency"]["threshold_ms"] == 100
    assert validation_config["tests"]["data_quality"]["missing_values"]["threshold"] == 0.1
    assert validation_config["tests"]["security"]["enabled"] == True

def test_rollback_setup(deployment_automation):
    rollback_config = deployment_automation.get_rollback_config()
    assert rollback_config is not None
    assert rollback_config["enabled"] == True
    assert rollback_config["strategy"] == "versioned"
    assert rollback_config["max_versions"] == 5
    assert rollback_config["backup"]["enabled"] == True
    assert rollback_config["backup"]["storage"] == "s3"
    assert rollback_config["backup"]["encryption"]["enabled"] == True

def test_monitoring_setup(deployment_automation):
    monitoring_config = deployment_automation.get_monitoring_config()
    assert monitoring_config is not None
    assert monitoring_config["enabled"] == True
    assert monitoring_config["metrics"]["performance"]["interval"] == "1h"
    assert monitoring_config["metrics"]["resource"]["interval"] == "5m"
    assert monitoring_config["alerts"]["email"]["recipients"] == ["admin@example.com"]

def test_logging_setup(deployment_automation):
    log_config = deployment_automation.get_logging_config()
    assert log_config is not None
    assert log_config["enabled"] == True
    assert log_config["level"] == "INFO"
    assert log_config["save_path"] == "./deployment/logs"
    assert log_config["rotation"]["max_size"] == "100MB"

def test_model_deployment_execution(deployment_automation, sample_model):
    results = deployment_automation.deploy_model(sample_model)
    assert results is not None
    assert isinstance(results, dict)
    assert "status" in results
    assert "version" in results
    assert "docker_container" in results
    assert "health_check" in results
    assert "scaling_status" in results
    assert "load_balancer" in results
    assert "security_status" in results
    assert "logging_status" in results
    assert "metrics_status" in results
    assert "backup_status" in results
    assert "disaster_recovery_status" in results

def test_model_validation_execution(deployment_automation, sample_data, sample_model):
    X, y = sample_data
    results = deployment_automation.validate_model(sample_model, X, y)
    assert results is not None
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "latency" in results
    assert "memory" in results
    assert "data_quality" in results
    assert "security" in results

def test_rollback_execution(deployment_automation):
    results = deployment_automation.rollback_model()
    assert results is not None
    assert isinstance(results, dict)
    assert "status" in results
    assert "previous_version" in results
    assert "backup_location" in results
    assert "encryption_status" in results

def test_monitoring_execution(deployment_automation):
    results = deployment_automation.monitor_deployment()
    assert results is not None
    assert isinstance(results, dict)
    assert "performance" in results
    assert "resource" in results
    assert "alerts" in results
    assert "health_status" in results

def test_deployment_performance(deployment_automation, sample_model):
    start_time = datetime.now()
    deployment_automation.deploy_model(sample_model)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    assert processing_time < 3600.0

def test_error_handling(deployment_automation):
    """
    오류 처리 테스트
    """
    try:
        # 오류 발생 시나리오 시뮬레이션
        result = deployment_automation.handle_error("테스트 오류")
        
        # 결과 검증
        assert result is not None
        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result
        assert "timestamp" in result
        
        # 로깅 확인
        assert deployment_automation.logger is not None
        assert deployment_automation.logger.level == logging.ERROR
        
    except Exception as e:
        # 예외 발생 시 로깅
        deployment_automation.logger.error(f"오류 처리 테스트 실패: {str(e)}")
        raise

def test_deployment_configuration():
    """배포 설정 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 배포 설정 조회
    config = automation.get_configuration()
    
    # 환경 설정 검증
    assert config["environment"]["enabled"] is True
    assert config["environment"]["settings"]["python_version"] == "3.8"
    assert config["environment"]["settings"]["dependencies"] == {
        "numpy": "1.21.0",
        "pandas": "1.3.0",
        "scikit-learn": "0.24.2"
    }
    
    # 검증 설정 검증
    assert config["validation"]["enabled"] is True
    assert config["validation"]["tests"]["accuracy"]["threshold"] == 0.8
    assert config["validation"]["tests"]["latency"]["threshold_ms"] == 100
    assert config["validation"]["tests"]["data_quality"]["missing_values"]["threshold"] == 0.1
    assert config["validation"]["tests"]["security"]["enabled"] is True
    
    # 롤백 설정 검증
    assert config["rollback"]["enabled"] is True
    assert config["rollback"]["strategy"] == "versioned"
    assert config["rollback"]["max_versions"] == 5
    assert config["rollback"]["backup"]["enabled"] is True
    assert config["rollback"]["backup"]["storage"] == "s3"
    assert config["rollback"]["backup"]["encryption"]["enabled"] is True
    
    # 모니터링 설정 검증
    assert config["monitoring"]["enabled"] is True
    assert config["monitoring"]["metrics"]["performance"]["interval"] == "1h"
    assert config["monitoring"]["metrics"]["resource"]["interval"] == "5m"
    assert config["monitoring"]["alerts"]["email"]["recipients"] == ["admin@example.com"]
    
    # 로깅 설정 검증
    assert config["logging"]["enabled"] is True
    assert config["logging"]["level"] == "INFO"
    assert config["logging"]["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert config["logging"]["save_path"] == "./deployment/logs"
    assert config["logging"]["rotation"]["max_size"] == "100MB"
    assert config["logging"]["rotation"]["backup_count"] == 5

def test_docker_container_status():
    """Docker 컨테이너 상태 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # Docker 컨테이너 상태 확인
    result = automation.deploy_docker(None)
    
    # 기본 상태 검증
    assert result["status"] == "running"
    assert result["ports"] == ["8000:8000"]
    assert result["environment"]["MODEL_ENV"] == "production"

def test_docker_health_check():
    """Docker 헬스 체크 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # Docker 헬스 체크 실행
    result = automation.deploy_docker(None)
    
    # 헬스 체크 검증
    assert result["health_check"]["status"] == "healthy"

def test_docker_scaling():
    """Docker 스케일링 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # Docker 스케일링 상태 확인
    result = automation.deploy_docker(None)
    
    # 스케일링 상태 검증
    assert result["scaling_status"]["current_replicas"] == 1
    assert result["scaling_status"]["target_replicas"] == 1

def test_docker_load_balancer():
    """Docker 로드 밸런서 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 로드 밸런서 상태 확인
    result = automation.deploy_docker(None)
    
    # 로드 밸런서 검증
    assert result["load_balancer"]["status"] == "active"
    assert result["load_balancer"]["endpoint"] == "http://localhost:8000"

def test_docker_security():
    """Docker 보안 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 보안 상태 확인
    result = automation.deploy_docker(None)
    
    # 보안 상태 검증
    assert result["security_status"]["ssl"] == "enabled"
    assert result["security_status"]["firewall"] == "active"

def test_docker_logging():
    """Docker 로깅 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 로깅 상태 확인
    result = automation.deploy_docker(None)
    
    # 로깅 상태 검증
    assert result["logging_status"]["level"] == "INFO"
    assert result["logging_status"]["path"] == "/var/log/deployment"

def test_docker_metrics():
    """Docker 메트릭스 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 메트릭스 상태 확인
    result = automation.deploy_docker(None)
    
    # 메트릭스 상태 검증
    assert result["metrics_status"]["prometheus"] == "active"
    assert result["metrics_status"]["grafana"] == "active"

def test_docker_backup():
    """Docker 백업 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 백업 상태 확인
    result = automation.deploy_docker(None)
    
    # 백업 상태 검증
    assert result["backup_status"]["status"] == "success"

def test_docker_disaster_recovery():
    """Docker 재해 복구 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 재해 복구 상태 확인
    result = automation.deploy_docker(None)
    
    # 재해 복구 상태 검증
    assert result["disaster_recovery_status"]["replication"] == "active"
    assert result["disaster_recovery_status"]["failover"] == "ready"

def test_backup_management():
    """백업 관리 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 백업 관리 실행
    result = automation.manage_backups()
    
    # 결과 검증
    assert result["backup_count"] == 10
    assert result["storage_used"] == 1000
    assert result["oldest_backup"] is not None
    assert result["encryption_status"] == "success"
    
    # 백업 스케줄링 검증
    schedule = automation.schedule_backup()
    assert schedule["schedule"] == "0 0 * * *"
    assert schedule["retention"] == 7
    assert schedule["storage"]["type"] == "local"
    assert schedule["storage"]["path"] == "./backups"
    assert schedule["last_backup"] is not None
    
    # 백업 보존 관리 검증
    retention = automation.manage_backup_retention()
    assert retention["total_backups"] == 10
    assert retention["retained_backups"] == 7
    assert retention["deleted_backups"] == 3
    assert retention["storage_cleaned"] == 300
    
    # 백업 암호화 검증
    encryption = automation.encrypt_backup()
    assert encryption["encryption_status"] == "success"
    assert encryption["algorithm"] == "AES-256"
    assert encryption["key_rotation"] is True
    assert encryption["integrity_check"] is True
    
    # 백업 복원 검증
    restore = automation.restore_backup()
    assert restore["restore_status"] == "success"
    assert restore["backup_version"] is not None
    assert restore["restore_time"] > 0
    assert restore["validation_status"] is True

def test_alert_management():
    """알림 관리 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 알림 전송
    result = automation._send_notification("테스트 알림", "테스트 메시지")
    
    # 결과 검증
    assert result["email_sent"] is True
    assert result["slack_sent"] is True
    assert result["status"] == "success"
    
    # 이메일 알림 검증
    email_result = automation._send_email_alert("테스트 이메일", "테스트 메시지")
    assert email_result is True
    
    # Slack 알림 검증
    slack_result = automation._send_slack_alert("테스트 Slack", "테스트 메시지")
    assert slack_result is True

def test_security_configuration():
    """보안 설정 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 보안 설정
    result = automation.configure_security()
    
    # SSL 상태 검증
    assert result["ssl_status"]["enabled"] is True
    assert result["ssl_status"]["valid"] is True
    assert result["ssl_status"]["expiration_days"] > 30
    
    # 방화벽 상태 검증
    assert result["firewall_status"]["enabled"] is True
    assert result["firewall_status"]["rules_count"] == 1
    assert result["firewall_status"]["default_policy"] == "deny"
    
    # 인증서 정보 검증
    assert result["certificate_info"]["issuer"] == "Let's Encrypt"
    assert result["certificate_info"]["subject"] == "model-service.example.com"
    assert result["certificate_info"]["valid_from"] is not None
    assert result["certificate_info"]["valid_to"] is not None
    
    # 접근 규칙 검증
    assert len(result["access_rules"]) == 1
    assert result["access_rules"][0]["port"] == 8000
    assert result["access_rules"][0]["protocol"] == "tcp"
    assert result["access_rules"][0]["action"] == "allow"
    assert result["access_rules"][0]["source"] == "0.0.0.0/0"

def test_health_check():
    """상태 확인 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 상태 확인 실행
    result = automation.check_health()
    
    # 결과 검증
    assert result["status"] == "healthy"
    assert result["checks"]["container"] == "healthy"
    assert result["checks"]["resource"] == "healthy"
    assert result["checks"]["network"] == "healthy"
    assert result["checks"]["logging"] == "healthy"
    assert result["checks"]["backup"] == "healthy"
    assert result["last_check"] is not None
    
    # 컨테이너 상태 확인
    container_status = automation._check_container_health()
    assert container_status == "healthy"
    
    # 리소스 상태 확인
    resource_status = automation._check_resource_health()
    assert resource_status == "healthy"
    
    # 네트워크 상태 확인
    network_status = automation._check_network_health()
    assert network_status == "healthy"
    
    # 로깅 상태 확인
    logging_status = automation._check_logging_health()
    assert logging_status == "healthy"
    
    # 백업 상태 확인
    backup_status = automation._check_backup_health()
    assert backup_status == "healthy"

def test_resource_monitoring():
    """리소스 모니터링 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 리소스 모니터링 실행
    result = automation.monitor_resources()
    
    # 결과 검증
    assert result["cpu_usage"] == 30.0
    assert result["memory_usage"] == 40.0
    assert result["disk_usage"] == 50.0
    assert result["network_usage"] == 20.0
    
    # CPU 사용량 모니터링
    cpu_usage = automation._monitor_cpu_usage()
    assert cpu_usage == 30.0
    
    # 메모리 사용량 모니터링
    memory_usage = automation._monitor_memory_usage()
    assert memory_usage == 40.0
    
    # 디스크 사용량 모니터링
    disk_usage = automation._monitor_disk_usage()
    assert disk_usage == 50.0
    
    # 네트워크 사용량 모니터링
    network_usage = automation._monitor_network_usage()
    assert network_usage == 20.0

def test_scaling_management():
    """스케일링 관리 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 스케일링 관리 실행
    result = automation.manage_scaling()
    
    # 결과 검증
    assert result["current_replicas"] == 2
    assert result["target_replicas"] == 3
    assert result["cpu_utilization"] == 30.0
    assert result["scaling_status"] == "scaling_out"
    
    # 현재 복제본 수 조회
    current_replicas = automation._get_current_replicas()
    assert current_replicas == 2
    
    # 목표 복제본 수 계산
    target_replicas = automation._calculate_target_replicas()
    assert target_replicas == 3
    
    # 스케일링 수행
    scaling_status = automation._perform_scaling(current_replicas, target_replicas)
    assert scaling_status == "scaling_out"

def test_version_management():
    """버전 관리 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 버전 관리 실행
    result = automation.manage_versions()
    
    # 결과 검증
    assert result["current_version"] == "1.0.0"
    assert result["available_versions"] == ["0.9.0", "1.0.0", "1.1.0"]
    assert result["latest_version"] == "1.1.0"
    assert len(result["version_history"]) == 2
    
    # 현재 버전 조회
    current_version = automation._get_current_version()
    assert current_version == "1.0.0"
    
    # 사용 가능한 버전 목록 조회
    available_versions = automation._get_available_versions()
    assert available_versions == ["0.9.0", "1.0.0", "1.1.0"]
    
    # 최신 버전 조회
    latest_version = automation._get_latest_version()
    assert latest_version == "1.1.0"
    
    # 버전 이력 조회
    version_history = automation._get_version_history()
    assert len(version_history) == 2
    assert version_history[0]["version"] == "1.1.0"
    assert version_history[1]["version"] == "1.0.0"

def test_dependency_management():
    """의존성 관리 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 의존성 관리 실행
    result = automation.manage_dependencies()
    
    # 결과 검증
    assert len(result["installed_packages"]) == 3
    assert len(result["required_packages"]) == 3
    assert result["version_matches"] is True
    assert result["update_available"] is True
    
    # 설치된 패키지 검증
    installed_packages = result["installed_packages"]
    assert installed_packages[0]["name"] == "numpy"
    assert installed_packages[0]["version"] == "1.21.0"
    assert installed_packages[1]["name"] == "pandas"
    assert installed_packages[1]["version"] == "1.3.0"
    assert installed_packages[2]["name"] == "scikit-learn"
    assert installed_packages[2]["version"] == "0.24.2"
    
    # 필요한 패키지 검증
    required_packages = result["required_packages"]
    assert required_packages[0]["name"] == "numpy"
    assert required_packages[0]["version"] == "1.21.0"
    assert required_packages[1]["name"] == "pandas"
    assert required_packages[1]["version"] == "1.3.0"
    assert required_packages[2]["name"] == "scikit-learn"
    assert required_packages[2]["version"] == "0.24.2"

def test_network_security():
    """네트워크 보안 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 네트워크 보안 상태 확인
    result = automation.check_network_security()
    
    # SSL 상태 검증
    assert result["ssl_status"]["enabled"] is True
    assert result["ssl_status"]["valid"] is True
    assert result["ssl_status"]["protocol"] == "TLSv1.3"
    assert result["ssl_status"]["cipher"] == "TLS_AES_256_GCM_SHA384"
    
    # 방화벽 상태 검증
    assert result["firewall_status"]["enabled"] is True
    assert result["firewall_status"]["rules_count"] == 1
    assert result["firewall_status"]["default_policy"] == "deny"
    
    # 포트 상태 검증
    assert len(result["ports"]) == 1
    port = result["ports"][0]
    assert port["number"] == 8000
    assert port["protocol"] == "tcp"
    assert port["state"] == "open"
    assert port["service"] == "model-service"
    
    # 연결 상태 검증
    assert len(result["connections"]) == 1
    connection = result["connections"][0]
    assert connection["source"] == "192.168.1.100"
    assert connection["destination"] == "model-service"
    assert connection["protocol"] == "tcp"
    assert connection["state"] == "established"
    assert connection["encrypted"] is True

def test_resource_optimization():
    """리소스 최적화 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 리소스 최적화 실행
    result = automation.optimize_resources()
    
    # CPU 사용량 검증
    assert result["cpu_usage"]["before"] > result["cpu_usage"]["after"]
    assert result["cpu_usage"]["threshold"] == 80.0
    assert result["cpu_usage"]["optimized"] is True
    
    # 메모리 사용량 검증
    assert result["memory_usage"]["before"] > result["memory_usage"]["after"]
    assert result["memory_usage"]["threshold"] == 80.0
    assert result["memory_usage"]["optimized"] is True
    
    # 네트워크 사용량 검증
    assert result["network_usage"]["before"] > result["network_usage"]["after"]
    assert result["network_usage"]["threshold"] == 80.0
    assert result["network_usage"]["optimized"] is True
    
    # 최적화 상태 검증
    assert result["optimization_status"]["success"] is True
    assert result["optimization_status"]["message"] == "Resources optimized successfully"
    assert result["optimization_status"]["timestamp"] is not None

def test_metrics_collection():
    """메트릭 수집 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 메트릭 수집
    result = automation.collect_metrics()
    
    # 성능 메트릭 검증
    assert result["performance"]["latency"] < 1000
    assert result["performance"]["throughput"] > 0
    assert result["performance"]["error_rate"] < 1.0
    assert result["performance"]["success_rate"] > 99.0
    
    # 리소스 메트릭 검증
    assert result["resource"]["cpu_usage"] < 80.0
    assert result["resource"]["memory_usage"] < 80.0
    assert result["resource"]["disk_usage"] < 80.0
    assert result["resource"]["network_usage"] < 80.0
    
    # 커스텀 메트릭 검증
    assert result["custom"]["model_accuracy"] > 0.8
    assert result["custom"]["model_version"] == "1.0.0"
    assert result["custom"]["deployment_status"] == "healthy"
    
    # 타임스탬프 검증
    assert result["timestamp"] is not None
    assert isinstance(result["timestamp"], str)
    assert len(result["timestamp"]) > 0

def test_prometheus_metrics():
    """Prometheus 메트릭 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # Prometheus 설정
    result = automation.configure_prometheus()
    
    # 기본 설정 검증
    assert result["enabled"] is True
    assert result["port"] == 9090
    assert result["path"] == "/metrics"
    
    # 메트릭 검증
    assert len(result["metrics"]) > 0
    metrics = result["metrics"]
    
    # 모델 성능 메트릭
    assert metrics["model_latency_seconds"]["type"] == "histogram"
    assert metrics["model_requests_total"]["type"] == "counter"
    assert metrics["model_errors_total"]["type"] == "counter"
    
    # 리소스 메트릭
    assert metrics["cpu_usage_percent"]["type"] == "gauge"
    assert metrics["memory_usage_bytes"]["type"] == "gauge"
    assert metrics["disk_usage_bytes"]["type"] == "gauge"
    
    # 커스텀 메트릭
    assert metrics["prediction_accuracy"]["type"] == "gauge"
    assert metrics["model_version"]["type"] == "gauge"
    assert metrics["deployment_status"]["type"] == "gauge"

def test_grafana_dashboard():
    """Grafana 대시보드 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # Grafana 설정
    result = automation.configure_grafana()
    
    # 기본 설정 검증
    assert result["enabled"] is True
    assert result["port"] == 3000
    assert result["dashboard"] == "model_deployment"
    
    # 패널 검증
    assert len(result["panels"]) >= 4
    panels = result["panels"]
    
    # 모델 성능 패널
    performance_panel = panels["model_performance"]
    assert performance_panel["title"] == "Model Performance"
    assert performance_panel["type"] == "graph"
    assert "latency" in performance_panel["metrics"]
    assert "throughput" in performance_panel["metrics"]
    
    # 리소스 사용량 패널
    resource_panel = panels["resource_usage"]
    assert resource_panel["title"] == "Resource Usage"
    assert resource_panel["type"] == "gauge"
    assert "cpu" in resource_panel["metrics"]
    assert "memory" in resource_panel["metrics"]
    
    # 오류율 패널
    error_panel = panels["error_rate"]
    assert error_panel["title"] == "Error Rate"
    assert error_panel["type"] == "stat"
    assert "errors" in error_panel["metrics"]
    
    # 배포 상태 패널
    deployment_panel = panels["deployment_status"]
    assert deployment_panel["title"] == "Deployment Status"
    assert deployment_panel["type"] == "status"
    assert "health" in deployment_panel["metrics"]

def test_metrics_visualization():
    """메트릭 시각화 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 메트릭 시각화
    result = automation.visualize_metrics()
    
    # 대시보드 URL 검증
    assert result["dashboard_url"] == "http://localhost:3000/d/model-deployment"
    
    # 패널 검증
    assert len(result["panels"]) >= 4
    panels = result["panels"]
    
    # 성능 패널 검증
    performance_panel = panels["performance"]
    assert performance_panel["title"] == "Performance Metrics"
    assert performance_panel["type"] == "graph"
    assert performance_panel["metrics"] == ["latency", "throughput", "error_rate"]
    
    # 리소스 패널 검증
    resource_panel = panels["resources"]
    assert resource_panel["title"] == "Resource Usage"
    assert resource_panel["type"] == "gauge"
    assert resource_panel["metrics"] == ["cpu", "memory", "disk", "network"]
    
    # 갱신 주기 검증
    assert result["refresh_interval"] == "5s"
    
    # 시간 범위 검증
    assert result["time_range"]["from"] == "now-1h"
    assert result["time_range"]["to"] == "now"

def test_scheduled_backup():
    """예약된 백업 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 백업 스케줄링
    result = automation.schedule_backup()
    
    # 스케줄 검증
    assert result["schedule"] == "0 0 * * *"  # 매일 자정
    assert result["retention"] == 7  # 7일간 보관
    
    # 저장소 설정 검증
    assert result["storage"]["type"] == "local"
    assert result["storage"]["path"] == "./backups"
    assert result["storage"]["max_size"] == "100GB"
    assert result["storage"]["compression"] is True
    
    # 마지막 백업 검증
    assert result["last_backup"] is not None
    assert result["last_backup"]["status"] == "success"
    assert result["last_backup"]["size"] > 0
    assert result["last_backup"]["timestamp"] is not None
    
    # 다음 백업 검증
    assert result["next_backup"] is not None
    assert result["next_backup"]["scheduled_time"] is not None
    assert result["next_backup"]["type"] == "full"

def test_backup_retention():
    """백업 보존 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 백업 보존 관리
    result = automation.manage_backup_retention()
    
    # 백업 수 검증
    assert result["total_backups"] == 10
    assert result["retained_backups"] == 7
    assert result["deleted_backups"] == 3
    
    # 저장소 정리 검증
    assert result["storage_cleaned"] == 300  # MB
    assert result["storage_saved"] == 30  # %
    
    # 보존 정책 검증
    assert result["retention_policy"]["days"] == 7
    assert result["retention_policy"]["max_backups"] == 10
    assert result["retention_policy"]["min_backups"] == 3
    
    # 백업 상태 검증
    assert len(result["backup_status"]) == 7
    backup = result["backup_status"][0]
    assert backup["id"] is not None
    assert backup["timestamp"] is not None
    assert backup["size"] > 0
    assert backup["type"] == "full"
    assert backup["status"] == "retained"

def test_backup_encryption():
    """백업 암호화 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 백업 암호화
    result = automation.encrypt_backup()
    
    # 암호화 상태 검증
    assert result["encryption_status"] == "success"
    assert result["algorithm"] == "AES-256"
    assert result["key_rotation"] is True
    assert result["integrity_check"] is True
    
    # 암호화 키 검증
    assert result["key_info"]["type"] == "symmetric"
    assert result["key_info"]["length"] == 256
    assert result["key_info"]["rotation_period"] == "30d"
    assert result["key_info"]["last_rotation"] is not None
    
    # 암호화된 파일 검증
    assert result["encrypted_files"] > 0
    assert result["encryption_time"] > 0
    assert result["total_size"] > 0
    
    # 무결성 검증
    assert result["integrity"]["checksum"] is not None
    assert result["integrity"]["verified"] is True
    assert result["integrity"]["last_check"] is not None

def test_backup_restore():
    """백업 복원 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 백업 복원
    result = automation.restore_backup()
    
    # 복원 상태 검증
    assert result["restore_status"] == "success"
    assert result["backup_version"] == "1.0.0"
    assert result["restore_time"] > 0
    assert result["validation_status"] is True
    
    # 복원된 파일 검증
    assert result["restored_files"] > 0
    assert result["total_size"] > 0
    assert result["checksum_verified"] is True
    
    # 복원 로그 검증
    assert len(result["restore_log"]) > 0
    log_entry = result["restore_log"][0]
    assert log_entry["timestamp"] is not None
    assert log_entry["action"] == "restore_started"
    assert log_entry["status"] == "success"
    
    # 복원 후 검증
    assert result["post_restore"]["model_status"] == "healthy"
    assert result["post_restore"]["config_status"] == "valid"
    assert result["post_restore"]["data_integrity"] is True

def test_disaster_recovery_setup():
    """재해 복구 설정 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 재해 복구 설정
    result = automation.configure_disaster_recovery()
    
    # 전략 검증
    assert result["strategy"] == "active-passive"
    assert result["status"] == "configured"
    
    # 복제 설정 검증
    assert result["replication"]["enabled"] is True
    assert result["replication"]["type"] == "synchronous"
    assert result["replication"]["interval"] == "1m"
    assert result["replication"]["target_location"] == "backup-site"
    
    # 장애 조치 설정 검증
    assert result["failover"]["enabled"] is True
    assert result["failover"]["automatic"] is True
    assert result["failover"]["threshold"] == "30s"
    assert result["failover"]["retry_count"] == 3
    
    # 복구 지점 설정 검증
    assert result["recovery_point"]["rpo"] == "1h"
    assert result["recovery_point"]["rto"] == "4h"
    assert result["recovery_point"]["backup_frequency"] == "1h"
    
    # 테스트 설정 검증
    assert result["testing"]["enabled"] is True
    assert result["testing"]["frequency"] == "1w"
    assert result["testing"]["last_test"] is not None
    assert result["testing"]["next_test"] is not None

def test_replication_management():
    """복제 관리 테스트"""
    automation = ModelDeploymentAutomation("config/model_deployment_automation.json")
    
    # 복제 관리 실행
    result = automation.manage_replication()
    
    # 기본 설정 검증
    assert result["status"] == "active"
    assert result["enabled"] is True
    assert result["type"] == "synchronous"
    
    # 복제 대상 검증
    assert result["target"]["location"] == "backup-site"
    assert result["target"]["status"] == "connected"
    assert result["target"]["last_sync"] is not None
    
    # 복제 성능 검증
    assert result["performance"]["latency"] < 100  # ms
    assert result["performance"]["bandwidth"] > 100  # MB/s
    assert result["performance"]["sync_time"] < 60  # seconds
    
    # 복제 상태 검증
    assert result["stats"]["total_files"] > 0
    assert result["stats"]["total_size"] > 0
    assert result["stats"]["success_rate"] > 99  # percentage
    
    # 모니터링 설정 검증
    assert result["monitoring"]["enabled"] is True
    assert result["monitoring"]["interval"] == "1m"
    assert result["monitoring"]["alerts_enabled"] is True
    
    # 로그 검증
    assert len(result["logs"]) > 0
    assert result["logs"][0]["timestamp"] is not None
    assert result["logs"][0]["action"] is not None
    assert result["logs"][0]["status"] == "success"

def test_failover_management(deployment_automation):
    results = deployment_automation.manage_failover()
    assert results is not None
    assert isinstance(results, dict)
    assert "enabled" in results
    assert "threshold" in results
    assert "timeout" in results
    assert "status" in results

def test_disaster_recovery_execution(deployment_automation):
    results = deployment_automation.execute_disaster_recovery()
    assert results is not None
    assert isinstance(results, dict)
    assert "recovery_status" in results
    assert "recovery_time" in results
    assert "data_integrity" in results
    assert "system_status" in results

def test_metrics_validation():
    """
    메트릭스 검증 테스트
    """
    automation = ModelDeploymentAutomation()
    metrics = automation._validate_metrics()
    
    # Prometheus 메트릭스 검증
    assert "prometheus" in metrics
    assert metrics["prometheus"]["scrape_interval"] == "15s"
    assert metrics["prometheus"]["retention"] == "15d"
    assert "model-service" in metrics["prometheus"]["targets"]
    assert "node-exporter" in metrics["prometheus"]["targets"]
    assert len(metrics["prometheus"]["rules"]["recording"]) > 0
    assert len(metrics["prometheus"]["rules"]["alerting"]) > 0
    
    # Grafana 대시보드 검증
    assert "grafana" in metrics
    assert "model_performance" in metrics["grafana"]["dashboards"]
    assert "system_metrics" in metrics["grafana"]["dashboards"]
    assert len(metrics["grafana"]["dashboards"]["model_performance"]["panels"]) > 0
    assert len(metrics["grafana"]["dashboards"]["system_metrics"]["panels"]) > 0
    
    # 커스텀 메트릭스 검증
    assert "custom_metrics" in metrics
    assert "model" in metrics["custom_metrics"]
    assert "latency" in metrics["custom_metrics"]["model"]
    assert "prediction_accuracy" in metrics["custom_metrics"]["model"]
    assert "request_rate" in metrics["custom_metrics"]["model"]
    
    # 시스템 메트릭스 검증
    assert "system" in metrics
    assert "cpu" in metrics["system"]
    assert "memory" in metrics["system"]
    assert "disk" in metrics["system"]
    assert all("usage" in metrics["system"][resource] for resource in ["cpu", "memory", "disk"])
    assert all("trend" in metrics["system"][resource] for resource in ["cpu", "memory", "disk"])
    assert all("threshold" in metrics["system"][resource] for resource in ["cpu", "memory", "disk"])
    
    # 비즈니스 메트릭스 검증
    assert "business" in metrics
    assert "throughput" in metrics["business"]
    assert "availability" in metrics["business"]
    assert "costs" in metrics["business"]
    assert all("current" in metrics["business"][metric] for metric in ["throughput", "availability"])
    assert all("trend" in metrics["business"][metric] for metric in ["throughput", "availability"])
    assert all("threshold" in metrics["business"][metric] for metric in ["throughput", "availability"])
    assert all(resource in metrics["business"]["costs"] for resource in ["compute", "storage", "network"])

def test_docker_deployment_setup():
    """Docker 배포 설정 테스트"""
    config_path = os.path.join("config", "model_deployment_automation.json")
    deployment = ModelDeploymentAutomation(config_path)
    config = deployment.get_configuration()
    assert config["environment"]["enabled"] is True
    assert config["environment"]["settings"]["python_version"] == "3.8"

def test_docker_container_creation():
    """Docker 컨테이너 생성 테스트"""
    config_path = os.path.join("config", "model_deployment_automation.json")
    deployment = ModelDeploymentAutomation(config_path)
    
    # Docker 컨테이너 생성 및 실행
    container = asyncio.run(deployment.deploy_docker("model_path"))
    
    # 결과 검증
    assert container["status"] == "running"
    assert container["ports"] == ["8000:8000"]
    assert container["environment"] == {"MODEL_ENV": "production"} 