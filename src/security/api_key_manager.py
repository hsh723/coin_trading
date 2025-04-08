import os
import json
import base64
from typing import Dict, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from ..utils.logger import setup_logger
from ..database.database import Database

class APIKeyManager:
    """API 키 관리 클래스"""
    
    def __init__(self, master_key: str):
        """
        API 키 관리 클래스 초기화
        
        Args:
            master_key (str): 마스터 키
        """
        self.logger = setup_logger('api_key_manager')
        self.db = Database()
        self.master_key = master_key.encode()
        self._initialize_encryption()
        
    def _initialize_encryption(self) -> None:
        """암호화 초기화"""
        try:
            # PBKDF2를 사용하여 마스터 키에서 암호화 키 생성
            salt = b'coin_trading_salt'  # 실제 운영에서는 랜덤한 salt 사용
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
            self.cipher_suite = Fernet(key)
            
        except Exception as e:
            self.logger.error(f"암호화 초기화 실패: {str(e)}")
            raise
            
    def save_api_key(self, exchange: str, api_key: str, api_secret: str) -> None:
        """
        API 키 저장
        
        Args:
            exchange (str): 거래소 이름
            api_key (str): API 키
            api_secret (str): API 시크릿
        """
        try:
            # API 키 암호화
            encrypted_key = self.cipher_suite.encrypt(api_key.encode())
            encrypted_secret = self.cipher_suite.encrypt(api_secret.encode())
            
            # 데이터베이스에 저장
            self.db.save_api_key({
                'exchange': exchange,
                'api_key': encrypted_key.decode(),
                'api_secret': encrypted_secret.decode(),
                'created_at': datetime.now(),
                'last_used': datetime.now()
            })
            
            self.logger.info(f"{exchange} API 키 저장 완료")
            
        except Exception as e:
            self.logger.error(f"API 키 저장 실패: {str(e)}")
            raise
            
    def get_api_key(self, exchange: str) -> Optional[Dict[str, str]]:
        """
        API 키 조회
        
        Args:
            exchange (str): 거래소 이름
            
        Returns:
            Optional[Dict[str, str]]: API 키 정보
        """
        try:
            # 데이터베이스에서 API 키 조회
            api_key_data = self.db.get_api_key(exchange)
            if not api_key_data:
                return None
                
            # API 키 복호화
            decrypted_key = self.cipher_suite.decrypt(api_key_data['api_key'].encode())
            decrypted_secret = self.cipher_suite.decrypt(api_key_data['api_secret'].encode())
            
            # 마지막 사용 시간 업데이트
            self.db.update_api_key_last_used(exchange)
            
            return {
                'api_key': decrypted_key.decode(),
                'api_secret': decrypted_secret.decode()
            }
            
        except Exception as e:
            self.logger.error(f"API 키 조회 실패: {str(e)}")
            return None
            
    def delete_api_key(self, exchange: str) -> None:
        """
        API 키 삭제
        
        Args:
            exchange (str): 거래소 이름
        """
        try:
            self.db.delete_api_key(exchange)
            self.logger.info(f"{exchange} API 키 삭제 완료")
            
        except Exception as e:
            self.logger.error(f"API 키 삭제 실패: {str(e)}")
            raise
            
    def rotate_api_key(self, exchange: str, new_api_key: str, new_api_secret: str) -> None:
        """
        API 키 교체
        
        Args:
            exchange (str): 거래소 이름
            new_api_key (str): 새로운 API 키
            new_api_secret (str): 새로운 API 시크릿
        """
        try:
            # 기존 API 키 삭제
            self.delete_api_key(exchange)
            
            # 새로운 API 키 저장
            self.save_api_key(exchange, new_api_key, new_api_secret)
            
            self.logger.info(f"{exchange} API 키 교체 완료")
            
        except Exception as e:
            self.logger.error(f"API 키 교체 실패: {str(e)}")
            raise
            
    def validate_api_key(self, exchange: str) -> bool:
        """
        API 키 유효성 검사
        
        Args:
            exchange (str): 거래소 이름
            
        Returns:
            bool: API 키 유효성 여부
        """
        try:
            api_key_data = self.db.get_api_key(exchange)
            if not api_key_data:
                return False
                
            # 마지막 사용 시간 확인
            last_used = api_key_data['last_used']
            if datetime.now() - last_used > timedelta(days=90):  # 90일 이상 미사용
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"API 키 유효성 검사 실패: {str(e)}")
            return False 