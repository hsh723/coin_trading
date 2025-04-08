"""
사용자 인증 모듈
"""

import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
import jwt
from pathlib import Path

class AuthManager:
    """인증 관리자 클래스"""
    
    def __init__(self, 
                secret_key: str,
                token_expiry: int = 3600,
                users_file: str = "users.json"):
        """
        초기화
        
        Args:
            secret_key (str): JWT 토큰 생성용 비밀 키
            token_expiry (int): 토큰 만료 시간 (초)
            users_file (str): 사용자 정보 파일 경로
        """
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.users_file = users_file
        self.logger = logging.getLogger(__name__)
        self._init_users_file()
        
    def _init_users_file(self):
        """사용자 정보 파일 초기화"""
        try:
            if not os.path.exists(self.users_file):
                with open(self.users_file, 'w') as f:
                    json.dump({}, f)
        except Exception as e:
            self.logger.error(f"사용자 정보 파일 초기화 실패: {str(e)}")
            raise
            
    def _load_users(self) -> Dict:
        """사용자 정보 로드"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"사용자 정보 로드 실패: {str(e)}")
            return {}
            
    def _save_users(self, users: Dict):
        """사용자 정보 저장"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=4)
        except Exception as e:
            self.logger.error(f"사용자 정보 저장 실패: {str(e)}")
            raise
            
    def _hash_password(self, password: str) -> str:
        """비밀번호 해시 생성"""
        salt = secrets.token_hex(16)
        return f"{salt}:{hashlib.sha256((password + salt).encode()).hexdigest()}"
        
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        try:
            salt, stored_hash = hashed_password.split(':')
            return hashlib.sha256((password + salt).encode()).hexdigest() == stored_hash
        except:
            return False
            
    def register_user(self, 
                     username: str,
                     password: str,
                     email: str,
                     settings: Optional[Dict] = None) -> bool:
        """
        사용자 등록
        
        Args:
            username (str): 사용자명
            password (str): 비밀번호
            email (str): 이메일
            settings (Optional[Dict]): 사용자 설정
            
        Returns:
            bool: 등록 성공 여부
        """
        try:
            users = self._load_users()
            
            if username in users:
                self.logger.warning(f"이미 존재하는 사용자명: {username}")
                return False
                
            users[username] = {
                'password': self._hash_password(password),
                'email': email,
                'settings': settings or {},
                'created_at': datetime.now().isoformat(),
                'last_login': None
            }
            
            self._save_users(users)
            self.logger.info(f"사용자 등록 완료: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"사용자 등록 실패: {str(e)}")
            return False
            
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        사용자 인증
        
        Args:
            username (str): 사용자명
            password (str): 비밀번호
            
        Returns:
            Optional[str]: JWT 토큰 또는 None
        """
        try:
            users = self._load_users()
            
            if username not in users:
                self.logger.warning(f"존재하지 않는 사용자명: {username}")
                return None
                
            if not self._verify_password(password, users[username]['password']):
                self.logger.warning(f"비밀번호 불일치: {username}")
                return None
                
            # 마지막 로그인 시간 업데이트
            users[username]['last_login'] = datetime.now().isoformat()
            self._save_users(users)
            
            # JWT 토큰 생성
            token = jwt.encode({
                'username': username,
                'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry)
            }, self.secret_key, algorithm='HS256')
            
            self.logger.info(f"사용자 인증 완료: {username}")
            return token
            
        except Exception as e:
            self.logger.error(f"사용자 인증 실패: {str(e)}")
            return None
            
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        토큰 검증
        
        Args:
            token (str): JWT 토큰
            
        Returns:
            Optional[Dict]: 토큰 페이로드 또는 None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("만료된 토큰")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("유효하지 않은 토큰")
            return None
            
    def update_user_settings(self, 
                           username: str,
                           settings: Dict) -> bool:
        """
        사용자 설정 업데이트
        
        Args:
            username (str): 사용자명
            settings (Dict): 새로운 설정
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            users = self._load_users()
            
            if username not in users:
                self.logger.warning(f"존재하지 않는 사용자명: {username}")
                return False
                
            users[username]['settings'].update(settings)
            self._save_users(users)
            
            self.logger.info(f"사용자 설정 업데이트 완료: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"사용자 설정 업데이트 실패: {str(e)}")
            return False
            
    def get_user_settings(self, username: str) -> Optional[Dict]:
        """
        사용자 설정 조회
        
        Args:
            username (str): 사용자명
            
        Returns:
            Optional[Dict]: 사용자 설정 또는 None
        """
        try:
            users = self._load_users()
            
            if username not in users:
                self.logger.warning(f"존재하지 않는 사용자명: {username}")
                return None
                
            return users[username]['settings']
            
        except Exception as e:
            self.logger.error(f"사용자 설정 조회 실패: {str(e)}")
            return None
            
    def change_password(self, 
                       username: str,
                       old_password: str,
                       new_password: str) -> bool:
        """
        비밀번호 변경
        
        Args:
            username (str): 사용자명
            old_password (str): 현재 비밀번호
            new_password (str): 새로운 비밀번호
            
        Returns:
            bool: 변경 성공 여부
        """
        try:
            users = self._load_users()
            
            if username not in users:
                self.logger.warning(f"존재하지 않는 사용자명: {username}")
                return False
                
            if not self._verify_password(old_password, users[username]['password']):
                self.logger.warning(f"현재 비밀번호 불일치: {username}")
                return False
                
            users[username]['password'] = self._hash_password(new_password)
            self._save_users(users)
            
            self.logger.info(f"비밀번호 변경 완료: {username}")
            return True
            
        except Exception as e:
            self.logger.error(f"비밀번호 변경 실패: {str(e)}")
            return False 