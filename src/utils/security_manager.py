"""
보안 관리 모듈
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import jwt
from cryptography.fernet import Fernet
import bcrypt
from functools import wraps
import hashlib
import hmac

class SecurityManager:
    """보안 관리 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.secret_key = os.getenv('SECRET_KEY', Fernet.generate_key())
        self.fernet = Fernet(self.secret_key)
        self.jwt_secret = os.getenv('JWT_SECRET', 'your-256-bit-secret')
        self.token_expiry = timedelta(days=1)
        
    def encrypt_api_key(self, api_key: str) -> str:
        """
        API 키 암호화
        
        Args:
            api_key (str): 암호화할 API 키
            
        Returns:
            str: 암호화된 API 키
        """
        try:
            return self.fernet.encrypt(api_key.encode()).decode()
            
        except Exception as e:
            self.logger.error(f"API 키 암호화 실패: {str(e)}")
            raise
            
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """
        API 키 복호화
        
        Args:
            encrypted_key (str): 복호화할 API 키
            
        Returns:
            str: 복호화된 API 키
        """
        try:
            return self.fernet.decrypt(encrypted_key.encode()).decode()
            
        except Exception as e:
            self.logger.error(f"API 키 복호화 실패: {str(e)}")
            raise
            
    def hash_password(self, password: str) -> str:
        """
        비밀번호 해시화
        
        Args:
            password (str): 해시화할 비밀번호
            
        Returns:
            str: 해시화된 비밀번호
        """
        try:
            salt = bcrypt.gensalt()
            return bcrypt.hashpw(password.encode(), salt).decode()
            
        except Exception as e:
            self.logger.error(f"비밀번호 해시화 실패: {str(e)}")
            raise
            
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        비밀번호 검증
        
        Args:
            password (str): 검증할 비밀번호
            hashed (str): 해시화된 비밀번호
            
        Returns:
            bool: 검증 결과
        """
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
            
        except Exception as e:
            self.logger.error(f"비밀번호 검증 실패: {str(e)}")
            return False
            
    def generate_token(self, user_id: str, permissions: list) -> str:
        """
        JWT 토큰 생성
        
        Args:
            user_id (str): 사용자 ID
            permissions (list): 권한 목록
            
        Returns:
            str: JWT 토큰
        """
        try:
            payload = {
                'user_id': user_id,
                'permissions': permissions,
                'exp': datetime.utcnow() + self.token_expiry
            }
            return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            
        except Exception as e:
            self.logger.error(f"토큰 생성 실패: {str(e)}")
            raise
            
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        JWT 토큰 검증
        
        Args:
            token (str): 검증할 토큰
            
        Returns:
            Optional[Dict]: 검증된 토큰 페이로드
        """
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("만료된 토큰")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("잘못된 토큰")
            return None
            
    def require_auth(self, required_permissions: list = None):
        """
        인증 데코레이터
        
        Args:
            required_permissions (list): 필요한 권한 목록
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # 토큰 검증
                    token = kwargs.get('token')
                    if not token:
                        raise ValueError("토큰이 필요합니다")
                        
                    payload = self.verify_token(token)
                    if not payload:
                        raise ValueError("유효하지 않은 토큰")
                        
                    # 권한 검증
                    if required_permissions:
                        user_permissions = payload.get('permissions', [])
                        if not all(p in user_permissions for p in required_permissions):
                            raise ValueError("권한이 없습니다")
                            
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    self.logger.error(f"인증 실패: {str(e)}")
                    raise
                    
            return wrapper
            
        return decorator
        
    def log_security_event(self, event_type: str, details: Dict):
        """
        보안 이벤트 로깅
        
        Args:
            event_type (str): 이벤트 유형
            details (Dict): 이벤트 상세 정보
        """
        try:
            event_data = {
                'timestamp': datetime.now(),
                'type': event_type,
                'details': details
            }
            self.db_manager.save_security_event(event_data)
            
        except Exception as e:
            self.logger.error(f"보안 이벤트 로깅 실패: {str(e)}")
            
    def check_access_control(self, user_id: str, resource: str) -> bool:
        """
        접근 제어 검사
        
        Args:
            user_id (str): 사용자 ID
            resource (str): 접근 리소스
            
        Returns:
            bool: 접근 허용 여부
        """
        try:
            # 사용자 권한 조회
            user_permissions = self.db_manager.get_user_permissions(user_id)
            
            # 리소스 접근 권한 확인
            resource_permissions = self.db_manager.get_resource_permissions(resource)
            
            return any(p in user_permissions for p in resource_permissions)
            
        except Exception as e:
            self.logger.error(f"접근 제어 검사 실패: {str(e)}")
            return False
            
    def generate_hmac(self, data: str) -> str:
        """
        HMAC 생성
        
        Args:
            data (str): HMAC 생성할 데이터
            
        Returns:
            str: HMAC 값
        """
        try:
            return hmac.new(
                self.secret_key.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
        except Exception as e:
            self.logger.error(f"HMAC 생성 실패: {str(e)}")
            raise
            
    def verify_hmac(self, data: str, hmac_value: str) -> bool:
        """
        HMAC 검증
        
        Args:
            data (str): 검증할 데이터
            hmac_value (str): HMAC 값
            
        Returns:
            bool: 검증 결과
        """
        try:
            expected_hmac = self.generate_hmac(data)
            return hmac.compare_digest(expected_hmac, hmac_value)
            
        except Exception as e:
            self.logger.error(f"HMAC 검증 실패: {str(e)}")
            return False 