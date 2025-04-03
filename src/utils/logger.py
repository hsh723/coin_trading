"""
로깅 시스템 모듈
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from .config import config_manager

class SensitiveDataFilter(logging.Filter):
    """민감 정보 필터링을 위한 로그 필터"""
    
    def filter(self, record):
        if hasattr(record, 'msg'):
            record.msg = config_manager.filter_sensitive_data(str(record.msg))
        if hasattr(record, 'args'):
            record.args = tuple(
                config_manager.filter_sensitive_data(str(arg))
                if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True

def setup_logger(name='trading_bot', level=logging.INFO):
    """로거 설정"""
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 설정 (날짜별)
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 민감 정보 필터 추가
    sensitive_filter = SensitiveDataFilter()
    file_handler.addFilter(sensitive_filter)
    console_handler.addFilter(sensitive_filter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 전역 로거 인스턴스
logger = setup_logger()

def get_logger(name: str) -> logging.Logger:
    """
    로거 가져오기
    
    Args:
        name (str): 로거 이름
        
    Returns:
        logging.Logger: 로거
    """
    return logging.getLogger(name) 