import os
from typing import Dict, Optional
import logging
from cryptography.fernet import Fernet
import base64

class APIKeyManager:
    """API 키 관리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encryption_key = None
        self.cipher_suite = None
        
    def initialize(self, encryption_key: Optional[str] = None) -> None:
        """초기화"""
        try:
            # 암호화 키 설정
            if encryption_key:
                self.encryption_key = encryption_key.encode()
            else:
                # 환경 변수에서 암호화 키 가져오기
                key = os.getenv('ENCRYPTION_KEY')
                if not key:
                    # 새로운 암호화 키 생성
                    key = Fernet.generate_key()
                    self.logger.warning("새로운 암호화 키가 생성되었습니다. .env 파일에 ENCRYPTION_KEY를 설정하세요.")
                self.encryption_key = key
                
            self.cipher_suite = Fernet(self.encryption_key)
            self.logger.info("API 키 관리자 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"API 키 관리자 초기화 중 오류 발생: {str(e)}")
            raise
            
    def encrypt(self, data: str) -> str:
        """데이터 암호화"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            self.logger.error(f"데이터 암호화 중 오류 발생: {str(e)}")
            raise
            
    def decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"데이터 복호화 중 오류 발생: {str(e)}")
            raise
            
    def get_api_keys(self) -> Dict[str, str]:
        """API 키 조회"""
        try:
            api_key = os.getenv('EXCHANGE_API_KEY')
            api_secret = os.getenv('EXCHANGE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("API 키가 설정되지 않았습니다.")
                
            # API 키 복호화
            decrypted_key = self.decrypt(api_key)
            decrypted_secret = self.decrypt(api_secret)
            
            return {
                'api_key': decrypted_key,
                'api_secret': decrypted_secret
            }
            
        except Exception as e:
            self.logger.error(f"API 키 조회 중 오류 발생: {str(e)}")
            raise 