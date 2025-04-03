"""
암호화 유틸리티 모듈
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class EncryptionManager:
    def __init__(self, key_file: str = '.env.key'):
        """
        암호화 관리자 초기화
        
        Args:
            key_file (str): 암호화 키 파일 경로
        """
        self.key_file = key_file
        self.key = self._load_or_generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def _load_or_generate_key(self) -> bytes:
        """
        암호화 키 로드 또는 생성
        
        Returns:
            bytes: 암호화 키
        """
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    return f.read()
            else:
                # 새로운 키 생성
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                return key
        except Exception as e:
            logger.error(f"암호화 키 로드/생성 실패: {str(e)}")
            raise
    
    def encrypt(self, data: str) -> str:
        """
        데이터 암호화
        
        Args:
            data (str): 암호화할 데이터
            
        Returns:
            str: 암호화된 데이터
        """
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"데이터 암호화 실패: {str(e)}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        데이터 복호화
        
        Args:
            encrypted_data (str): 복호화할 데이터
            
        Returns:
            str: 복호화된 데이터
        """
        try:
            decrypted_data = self.cipher_suite.decrypt(base64.b64decode(encrypted_data))
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"데이터 복호화 실패: {str(e)}")
            raise
    
    def encrypt_file(self, file_path: str) -> None:
        """
        파일 암호화
        
        Args:
            file_path (str): 암호화할 파일 경로
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.cipher_suite.encrypt(data)
            
            with open(f"{file_path}.enc", 'wb') as f:
                f.write(encrypted_data)
                
            logger.info(f"파일 암호화 완료: {file_path}")
        except Exception as e:
            logger.error(f"파일 암호화 실패: {str(e)}")
            raise
    
    def decrypt_file(self, encrypted_file_path: str) -> None:
        """
        파일 복호화
        
        Args:
            encrypted_file_path (str): 복호화할 파일 경로
        """
        try:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            output_path = encrypted_file_path[:-4]  # .enc 제거
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
                
            logger.info(f"파일 복호화 완료: {encrypted_file_path}")
        except Exception as e:
            logger.error(f"파일 복호화 실패: {str(e)}")
            raise

# 전역 암호화 관리자 인스턴스
encryption_manager = EncryptionManager() 