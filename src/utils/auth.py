"""
사용자 인증 모듈
"""

import os
import jwt
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from .config import config_manager
from .logger import logger
from .database import DatabaseManager

class AuthManager:
    def __init__(self):
        """인증 관리자 초기화"""
        self.db = DatabaseManager()
        self.jwt_secret = config_manager.get_config('JWT_SECRET', os.urandom(32).hex())
        self.jwt_expiry = 3600  # 1시간
        self.reauth_timeout = 300  # 5분
        self._setup_admin_user()
    
    def _setup_admin_user(self):
        """관리자 사용자 설정"""
        try:
            # 기본 관리자 계정이 없으면 생성
            if not self.db.get_user('admin'):
                self.create_user(
                    username='admin',
                    password='admin123',  # 초기 비밀번호
                    role='admin'
                )
                logger.warning("기본 관리자 계정이 생성되었습니다. 비밀번호를 변경하세요.")
        except Exception as e:
            logger.error(f"관리자 계정 설정 실패: {str(e)}")
    
    def create_user(self, username: str, password: str, role: str = 'user') -> bool:
        """
        사용자 생성
        
        Args:
            username (str): 사용자 이름
            password (str): 비밀번호
            role (str): 역할 (admin/user)
            
        Returns:
            bool: 생성 성공 여부
        """
        try:
            # 비밀번호 해시화
            hashed_password = self._hash_password(password)
            
            # 사용자 정보 저장
            user_data = {
                'username': username,
                'password': hashed_password,
                'role': role,
                'created_at': datetime.now(),
                'last_login': None
            }
            
            return self.db.save_user(user_data)
            
        except Exception as e:
            logger.error(f"사용자 생성 실패: {str(e)}")
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        사용자 인증
        
        Args:
            username (str): 사용자 이름
            password (str): 비밀번호
            
        Returns:
            Optional[str]: JWT 토큰 또는 None
        """
        try:
            # 사용자 조회
            user = self.db.get_user(username)
            if not user:
                logger.warning(f"사용자를 찾을 수 없음: {username}")
                return None
            
            # 비밀번호 확인
            if not self._verify_password(password, user['password']):
                logger.warning(f"잘못된 비밀번호: {username}")
                return None
            
            # 마지막 로그인 시간 업데이트
            self.db.update_user_last_login(username)
            
            # JWT 토큰 생성
            token = self._generate_token(user)
            return token
            
        except Exception as e:
            logger.error(f"인증 실패: {str(e)}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        토큰 검증
        
        Args:
            token (str): JWT 토큰
            
        Returns:
            Optional[Dict[str, Any]]: 사용자 정보 또는 None
        """
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("만료된 토큰")
            return None
        except jwt.InvalidTokenError:
            logger.warning("잘못된 토큰")
            return None
        except Exception as e:
            logger.error(f"토큰 검증 실패: {str(e)}")
            return None
    
    def require_reauth(self, token: str) -> bool:
        """
        재인증 필요 여부 확인
        
        Args:
            token (str): JWT 토큰
            
        Returns:
            bool: 재인증 필요 여부
        """
        try:
            payload = self.verify_token(token)
            if not payload:
                return True
            
            # 마지막 인증 시간 확인
            last_auth = payload.get('last_auth', 0)
            current_time = time.time()
            
            return (current_time - last_auth) > self.reauth_timeout
            
        except Exception as e:
            logger.error(f"재인증 확인 실패: {str(e)}")
            return True
    
    def _hash_password(self, password: str) -> str:
        """비밀번호 해시화"""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return salt.hex() + key.hex()
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """비밀번호 검증"""
        salt = bytes.fromhex(hashed[:64])
        key = bytes.fromhex(hashed[64:])
        new_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return key == new_key
    
    def _generate_token(self, user: Dict[str, Any]) -> str:
        """JWT 토큰 생성"""
        payload = {
            'username': user['username'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(seconds=self.jwt_expiry),
            'iat': datetime.utcnow(),
            'last_auth': time.time()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

# 전역 인증 관리자 인스턴스
auth_manager = AuthManager() 