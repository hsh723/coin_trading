import pytest
from src.logger.logger_manager import LoggerManager
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

@pytest.fixture
def logger_manager():
    config_dir = "./config"
    log_dir = "./logs"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "logger": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": {
                    "file": {
                        "filename": "./logs/trading.log",
                        "max_bytes": 10485760,
                        "backup_count": 5
                    },
                    "console": {
                        "enabled": True
                    }
                }
            }
        }
    }
    with open(os.path.join(config_dir, "logger.json"), "w") as f:
        json.dump(config, f)
    
    return LoggerManager(config_dir=config_dir, log_dir=log_dir)

def test_logger_manager_initialization(logger_manager):
    assert logger_manager is not None
    assert logger_manager.config_dir == "./config"
    assert logger_manager.log_dir == "./logs"

def test_logger_levels(logger_manager):
    # 로그 레벨 테스트
    logger = logger_manager.get_logger("test")
    
    # DEBUG 레벨
    logger.debug("Debug message")
    
    # INFO 레벨
    logger.info("Info message")
    
    # WARNING 레벨
    logger.warning("Warning message")
    
    # ERROR 레벨
    logger.error("Error message")
    
    # CRITICAL 레벨
    logger.critical("Critical message")

def test_logger_handlers(logger_manager):
    # 로그 핸들러 테스트
    logger = logger_manager.get_logger("test")
    
    # 핸들러 확인
    handlers = logger.handlers
    assert len(handlers) >= 1
    
    # 파일 핸들러 확인
    file_handler = None
    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break
    assert file_handler is not None
    
    # 콘솔 핸들러 확인
    console_handler = None
    for handler in handlers:
        if isinstance(handler, logging.StreamHandler):
            console_handler = handler
            break
    assert console_handler is not None

def test_logger_format(logger_manager):
    # 로그 포맷 테스트
    logger = logger_manager.get_logger("test")
    
    # 로그 메시지 생성
    test_message = "Test log message"
    logger.info(test_message)
    
    # 로그 파일 확인
    log_file = os.path.join(logger_manager.log_dir, "trading.log")
    with open(log_file, "r") as f:
        log_content = f.read()
        assert test_message in log_content
        assert "INFO" in log_content
        assert "test" in log_content

def test_logger_rotation(logger_manager):
    # 로그 파일 로테이션 테스트
    logger = logger_manager.get_logger("test")
    
    # 대량의 로그 메시지 생성
    for i in range(1000):
        logger.info(f"Test log message {i}")
    
    # 로그 파일 확인
    log_files = os.listdir(logger_manager.log_dir)
    assert len(log_files) >= 1

def test_logger_performance(logger_manager):
    # 로거 성능 테스트
    logger = logger_manager.get_logger("test")
    
    # 대량의 로그 메시지 처리
    start_time = datetime.now()
    
    for i in range(1000):
        logger.info(f"Performance test message {i}")
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1000개의 로그를 1초 이내에 처리
    assert processing_time < 1.0

def test_logger_error_handling(logger_manager):
    # 로거 에러 처리 테스트
    logger = logger_manager.get_logger("test")
    
    # 잘못된 로그 레벨
    with pytest.raises(ValueError):
        logger.log("INVALID_LEVEL", "Invalid level message")
    
    # 잘못된 로그 포맷
    with pytest.raises(ValueError):
        logger_manager.set_format("Invalid format %(invalid)s")

def test_logger_configuration(logger_manager):
    # 로거 설정 테스트
    config = logger_manager.get_configuration()
    
    assert config is not None
    assert "logger" in config
    assert "level" in config["logger"]
    assert "format" in config["logger"]
    assert "handlers" in config["logger"]
    assert "file" in config["logger"]["handlers"]
    assert "console" in config["logger"]["handlers"]

def test_logger_custom_handler(logger_manager):
    # 커스텀 핸들러 테스트
    # 메모리 핸들러 생성
    memory_handler = logging.handlers.MemoryHandler(capacity=100)
    
    # 커스텀 핸들러 추가
    logger_manager.add_handler("memory", memory_handler)
    
    # 로거 가져오기
    logger = logger_manager.get_logger("test")
    
    # 로그 메시지 생성
    test_message = "Custom handler test message"
    logger.info(test_message)
    
    # 메모리 핸들러 확인
    assert test_message in str(memory_handler.buffer)

def test_logger_context(logger_manager):
    # 로거 컨텍스트 테스트
    with logger_manager.get_logger("test") as logger:
        logger.info("Context test message")
        
        # 컨텍스트 내에서 로그 레벨 변경
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug message in context")
    
    # 컨텍스트 종료 후 로그 레벨 복원 확인
    logger = logger_manager.get_logger("test")
    assert logger.level == logging.INFO 