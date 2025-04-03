"""
환경 변수 및 설정 관리 모듈
"""

import os
from dotenv import load_dotenv
import yaml
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        """설정 관리자 초기화"""
        self.env_file = Path('.env')
        self.config_file = Path('config.yaml')
        self._load_env()
        self._load_config()
    
    def _load_env(self):
        """환경 변수 로드"""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info("환경 변수 파일 로드 완료")
        else:
            logger.warning("환경 변수 파일이 존재하지 않습니다.")
    
    def _load_config(self):
        """설정 파일 로드"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info("설정 파일 로드 완료")
        else:
            self.config = {}
            logger.warning("설정 파일이 존재하지 않습니다.")
    
    def get_api_key(self, key_name):
        """API 키 조회 (마스킹 처리)"""
        key = os.getenv(key_name)
        if key:
            return self._mask_sensitive_data(key)
        return None
    
    def get_config(self, key, default=None):
        """설정값 조회"""
        return self.config.get(key, default)
    
    def update_config(self, key, value, require_confirmation=True):
        """설정값 업데이트"""
        try:
            if require_confirmation:
                # 중요 설정 변경 시 확인
                if self._is_sensitive_config(key):
                    if not self._confirm_config_change(key, value):
                        logger.warning(f"설정 변경 취소됨: {key}")
                        return False
            
            # 설정 파일 업데이트
            self.config[key] = value
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"설정 업데이트 완료: {key}")
            return True
        except Exception as e:
            logger.error(f"설정 업데이트 실패: {str(e)}")
            return False
    
    def _is_sensitive_config(self, key):
        """민감한 설정 여부 확인"""
        sensitive_keys = [
            'api_key', 'secret', 'password', 'token',
            'access_key', 'private_key', 'credentials'
        ]
        return any(sensitive in key.lower() for sensitive in sensitive_keys)
    
    def _confirm_config_change(self, key, value):
        """설정 변경 확인"""
        # 실제 구현에서는 사용자 인터페이스에서 확인 받아야 함
        logger.warning(f"중요 설정 변경 시도: {key}")
        logger.warning(f"변경 전 값: {self.get_config(key)}")
        logger.warning(f"변경 후 값: {self._mask_sensitive_data(value)}")
        return True  # 임시로 항상 True 반환
    
    def _mask_sensitive_data(self, data):
        """민감 정보 마스킹 처리"""
        if not data:
            return data
        
        if isinstance(data, str):
            if len(data) <= 4:
                return '*' * len(data)
            return data[:2] + '*' * (len(data) - 4) + data[-2:]
        return data
    
    def filter_sensitive_data(self, message):
        """로그 메시지에서 민감 정보 필터링"""
        if not message:
            return message
        
        # API 키 패턴
        patterns = [
            r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
            r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
            r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
            r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?'
        ]
        
        for pattern in patterns:
            message = re.sub(
                pattern,
                lambda m: f"{m.group(0).split('=')[0]}={self._mask_sensitive_data(m.group(1))}",
                message,
                flags=re.IGNORECASE
            )
        
        return message

# 전역 설정 관리자 인스턴스
config_manager = ConfigManager() 