import pytest
from src.utils.logging import LogManager
import os
import json
import time
import logging
from datetime import datetime

@pytest.fixture
def log_manager():
    config_dir = "./config"
    data_dir = "./data"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "log_level": "INFO",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_rotation": {
                "enabled": True,
                "max_size": 10485760,  # 10MB
                "backup_count": 5
            },
            "log_types": {
                "trading": {
                    "enabled": True,
                    "file": "trading.log"
                },
                "system": {
                    "enabled": True,
                    "file": "system.log"
                },
                "error": {
                    "enabled": True,
                    "file": "error.log"
                }
            }
        }
    }
    with open(os.path.join(config_dir, "logging.json"), "w") as f:
        json.dump(config, f)
    
    return LogManager(config_dir=config_dir, data_dir=data_dir)

def test_log_manager_initialization(log_manager):
    assert log_manager is not None
    assert log_manager.config_dir == "./config"
    assert log_manager.data_dir == "./data"

def test_log_levels(log_manager):
    # 로그 레벨 설정 테스트
    log_manager.set_log_level("DEBUG")
    assert log_manager.get_log_level() == "DEBUG"
    
    log_manager.set_log_level("INFO")
    assert log_manager.get_log_level() == "INFO"
    
    log_manager.set_log_level("WARNING")
    assert log_manager.get_log_level() == "WARNING"
    
    log_manager.set_log_level("ERROR")
    assert log_manager.get_log_level() == "ERROR"

def test_log_format(log_manager):
    # 로그 포맷 설정 테스트
    new_format = "%(levelname)s - %(message)s"
    log_manager.set_log_format(new_format)
    assert log_manager.get_log_format() == new_format

def test_log_rotation(log_manager):
    # 로그 로테이션 설정 테스트
    rotation_config = {
        "enabled": True,
        "max_size": 5242880,  # 5MB
        "backup_count": 3
    }
    log_manager.set_log_rotation(rotation_config)
    assert log_manager.get_log_rotation() == rotation_config

def test_log_types(log_manager):
    # 로그 타입 설정 테스트
    log_types = log_manager.get_log_types()
    assert "trading" in log_types
    assert "system" in log_types
    assert "error" in log_types
    
    # 새로운 로그 타입 추가
    log_manager.add_log_type("performance", {
        "enabled": True,
        "file": "performance.log"
    })
    assert "performance" in log_manager.get_log_types()

def test_log_writing(log_manager):
    # 로그 작성 테스트
    test_message = "테스트 로그 메시지"
    log_manager.log("INFO", test_message)
    
    # 로그 파일 확인
    log_file = os.path.join(log_manager.data_dir, "system.log")
    assert os.path.exists(log_file)
    
    with open(log_file, "r") as f:
        log_content = f.read()
        assert test_message in log_content

def test_log_retrieval(log_manager):
    # 로그 검색 테스트
    test_message = "검색 테스트 메시지"
    log_manager.log("INFO", test_message)
    
    # 로그 검색
    logs = log_manager.search_logs("검색 테스트")
    assert len(logs) > 0
    assert any(test_message in log for log in logs)

def test_log_cleaning(log_manager):
    # 로그 정리 테스트
    # 테스트 로그 작성
    for i in range(100):
        log_manager.log("INFO", f"테스트 로그 {i}")
    
    # 로그 정리
    log_manager.clean_logs(max_age_days=0)
    
    # 로그 파일 확인
    log_file = os.path.join(log_manager.data_dir, "system.log")
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) == 0

def test_error_logging(log_manager):
    # 에러 로깅 테스트
    try:
        raise ValueError("테스트 에러")
    except Exception as e:
        log_manager.log_error(e)
    
    # 에러 로그 파일 확인
    error_log_file = os.path.join(log_manager.data_dir, "error.log")
    assert os.path.exists(error_log_file)
    
    with open(error_log_file, "r") as f:
        log_content = f.read()
        assert "ValueError" in log_content
        assert "테스트 에러" in log_content

def test_performance_logging(log_manager):
    # 성능 로깅 테스트
    start_time = time.time()
    time.sleep(0.1)  # 100ms 지연
    end_time = time.time()
    
    log_manager.log_performance("test_operation", start_time, end_time)
    
    # 성능 로그 파일 확인
    perf_log_file = os.path.join(log_manager.data_dir, "performance.log")
    assert os.path.exists(perf_log_file)
    
    with open(perf_log_file, "r") as f:
        log_content = f.read()
        assert "test_operation" in log_content
        assert "latency" in log_content

def test_log_statistics(log_manager):
    # 로그 통계 테스트
    # 다양한 레벨의 로그 작성
    log_manager.log("INFO", "정보 로그")
    log_manager.log("WARNING", "경고 로그")
    log_manager.log("ERROR", "에러 로그")
    
    # 통계 확인
    stats = log_manager.get_log_statistics()
    assert stats is not None
    assert "total_logs" in stats
    assert "log_levels" in stats
    assert stats["log_levels"]["INFO"] > 0
    assert stats["log_levels"]["WARNING"] > 0
    assert stats["log_levels"]["ERROR"] > 0 