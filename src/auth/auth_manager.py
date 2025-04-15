import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import hashlib
import secrets
import jwt
import bcrypt
import logging
import threading
import queue
import time
import pandas as pd
from pathlib import Path

class AuthManager:
    """인증 관리 클래스"""
    
    def __init__(self,
                 data_dir: str = "./data",
                 config_dir: str = "./config"):
        """
        인증 관리자 초기화
        
        Args:
            data_dir: 데이터 디렉토리
            config_dir: 설정 디렉토리
        """
        self.data_dir = data_dir
        self.config_dir = config_dir
        
        # 로거 설정
        self.logger = logging.getLogger("auth_manager")
        
        # 세션 큐
        self.session_queue = queue.Queue()
        
        # 데이터베이스 연결
        self.db_path = os.path.join(data_dir, "auth.db")
        self._init_database()
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        # 설정 로드
        self.config = self._load_config()
        
        # 세션 관리 스레드
        self.session_thread = None
        self.is_running = False
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            config_path = os.path.join(self.config_dir, "auth_config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {
                "jwt_secret": secrets.token_hex(32),
                "jwt_expiry": 3600,
                "password_salt_rounds": 12,
                "session_timeout": 1800
            }
            
    def _init_database(self) -> None:
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 사용자 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    email TEXT UNIQUE,
                    role TEXT,
                    is_active BOOLEAN,
                    created_at DATETIME,
                    updated_at DATETIME
                )
            ''')
            
            # 역할 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS roles (
                    name TEXT PRIMARY KEY,
                    permissions TEXT,
                    created_at DATETIME,
                    updated_at DATETIME
                )
            ''')
            
            # 세션 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER,
                    expires_at DATETIME,
                    created_at DATETIME,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 기본 역할 생성
            cursor.execute('''
                INSERT OR IGNORE INTO roles
                (name, permissions, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (
                "admin",
                json.dumps(["read", "write", "delete", "admin"]),
                datetime.now(),
                datetime.now()
            ))
            
            cursor.execute('''
                INSERT OR IGNORE INTO roles
                (name, permissions, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (
                "user",
                json.dumps(["read", "write"]),
                datetime.now(),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 중 오류 발생: {e}")
            raise
            
    def start_session_management(self) -> None:
        """세션 관리 시작"""
        try:
            self.is_running = True
            self.session_thread = threading.Thread(target=self._manage_sessions)
            self.session_thread.start()
            
        except Exception as e:
            self.logger.error(f"세션 관리 시작 중 오류 발생: {e}")
            raise
            
    def stop_session_management(self) -> None:
        """세션 관리 중지"""
        try:
            self.is_running = False
            if self.session_thread:
                self.session_thread.join()
                
        except Exception as e:
            self.logger.error(f"세션 관리 중지 중 오류 발생: {e}")
            raise
            
    def _manage_sessions(self) -> None:
        """세션 관리 루프"""
        try:
            while self.is_running:
                # 만료된 세션 정리
                self._cleanup_expired_sessions()
                
                # 세션 큐 처리
                if not self.session_queue.empty():
                    session = self.session_queue.get()
                    self._update_session(session)
                    
                time.sleep(60)
                
        except Exception as e:
            self.logger.error(f"세션 관리 중 오류 발생: {e}")
            
    def _cleanup_expired_sessions(self) -> None:
        """만료된 세션 정리"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM sessions
                WHERE expires_at < ?
            ''', (datetime.now(),))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"세션 정리 중 오류 발생: {e}")
            
    def _update_session(self, session: Dict[str, Any]) -> None:
        """
        세션 업데이트
        
        Args:
            session: 세션 정보
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions
                SET expires_at = ?
                WHERE token = ?
            ''', (
                datetime.now() + timedelta(seconds=self.config["session_timeout"]),
                session["token"]
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"세션 업데이트 중 오류 발생: {e}")
            
    def register_user(self,
                     username: str,
                     password: str,
                     email: str,
                     role: str = "user") -> Dict[str, Any]:
        """
        사용자 등록
        
        Args:
            username: 사용자 이름
            password: 비밀번호
            email: 이메일
            role: 역할
            
        Returns:
            사용자 정보
        """
        try:
            # 비밀번호 해시
            password_hash = bcrypt.hashpw(
                password.encode(),
                bcrypt.gensalt(self.config["password_salt_rounds"])
            ).decode()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 사용자 생성
            cursor.execute('''
                INSERT INTO users
                (username, password, email, role, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                username,
                password_hash,
                email,
                role,
                True,
                datetime.now(),
                datetime.now()
            ))
            
            user_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return {
                "id": user_id,
                "username": username,
                "email": email,
                "role": role,
                "is_active": True
            }
            
        except Exception as e:
            self.logger.error(f"사용자 등록 중 오류 발생: {e}")
            raise
            
    def authenticate_user(self,
                         username: str,
                         password: str) -> Optional[Dict[str, Any]]:
        """
        사용자 인증
        
        Args:
            username: 사용자 이름
            password: 비밀번호
            
        Returns:
            인증 토큰
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 사용자 조회
            cursor.execute('''
                SELECT id, username, password, role, is_active
                FROM users
                WHERE username = ?
            ''', (username,))
            
            result = cursor.fetchone()
            
            if not result:
                return None
                
            user_id, username, password_hash, role, is_active = result
            
            if not is_active:
                return None
                
            # 비밀번호 검증
            if not bcrypt.checkpw(password.encode(), password_hash.encode()):
                return None
                
            # JWT 토큰 생성
            token = jwt.encode({
                "user_id": user_id,
                "username": username,
                "role": role,
                "exp": datetime.now() + timedelta(seconds=self.config["jwt_expiry"])
            }, self.config["jwt_secret"], algorithm="HS256")
            
            # 세션 저장
            cursor.execute('''
                INSERT INTO sessions
                (token, user_id, expires_at, created_at)
                VALUES (?, ?, ?, ?)
            ''', (
                token,
                user_id,
                datetime.now() + timedelta(seconds=self.config["session_timeout"]),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            return {
                "token": token,
                "user_id": user_id,
                "username": username,
                "role": role
            }
            
        except Exception as e:
            self.logger.error(f"사용자 인증 중 오류 발생: {e}")
            raise
            
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        토큰 검증
        
        Args:
            token: JWT 토큰
            
        Returns:
            토큰 정보
        """
        try:
            # 토큰 디코딩
            payload = jwt.decode(token, self.config["jwt_secret"], algorithms=["HS256"])
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 세션 검증
            cursor.execute('''
                SELECT user_id, expires_at
                FROM sessions
                WHERE token = ?
            ''', (token,))
            
            result = cursor.fetchone()
            
            if not result:
                return None
                
            user_id, expires_at = result
            
            if datetime.now() > datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S"):
                return None
                
            # 세션 갱신
            self.session_queue.put({
                "token": token,
                "expires_at": datetime.now() + timedelta(seconds=self.config["session_timeout"])
            })
            
            return payload
            
        except Exception as e:
            self.logger.error(f"토큰 검증 중 오류 발생: {e}")
            return None
            
    def logout(self, token: str) -> None:
        """
        로그아웃
        
        Args:
            token: JWT 토큰
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM sessions
                WHERE token = ?
            ''', (token,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"로그아웃 중 오류 발생: {e}")
            raise
            
    def check_permission(self,
                        token: str,
                        permission: str) -> bool:
        """
        권한 확인
        
        Args:
            token: JWT 토큰
            permission: 권한
            
        Returns:
            권한 여부
        """
        try:
            # 토큰 검증
            payload = self.validate_token(token)
            
            if not payload:
                return False
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 역할 권한 조회
            cursor.execute('''
                SELECT permissions
                FROM roles
                WHERE name = ?
            ''', (payload["role"],))
            
            result = cursor.fetchone()
            
            if not result:
                return False
                
            permissions = json.loads(result[0])
            
            return permission in permissions
            
        except Exception as e:
            self.logger.error(f"권한 확인 중 오류 발생: {e}")
            return False
            
    def get_user_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        사용자 정보 조회
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            사용자 정보
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, role, is_active, created_at, updated_at
                FROM users
                WHERE id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return None
                
            return {
                "id": result[0],
                "username": result[1],
                "email": result[2],
                "role": result[3],
                "is_active": result[4],
                "created_at": result[5],
                "updated_at": result[6]
            }
            
        except Exception as e:
            self.logger.error(f"사용자 정보 조회 중 오류 발생: {e}")
            raise
            
    def update_user(self,
                    user_id: int,
                    **kwargs) -> Optional[Dict[str, Any]]:
        """
        사용자 정보 업데이트
        
        Args:
            user_id: 사용자 ID
            **kwargs: 업데이트할 필드
            
        Returns:
            업데이트된 사용자 정보
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 업데이트할 필드 생성
            update_fields = []
            params = []
            
            for key, value in kwargs.items():
                if key == "password":
                    value = bcrypt.hashpw(
                        value.encode(),
                        bcrypt.gensalt(self.config["password_salt_rounds"])
                    ).decode()
                    
                update_fields.append(f"{key} = ?")
                params.append(value)
                
            params.append(user_id)
            
            # 사용자 정보 업데이트
            cursor.execute(f'''
                UPDATE users
                SET {', '.join(update_fields)}, updated_at = ?
                WHERE id = ?
            ''', (*params, datetime.now(), user_id))
            
            conn.commit()
            conn.close()
            
            return self.get_user_info(user_id)
            
        except Exception as e:
            self.logger.error(f"사용자 정보 업데이트 중 오류 발생: {e}")
            raise
            
    def delete_user(self, user_id: int) -> bool:
        """
        사용자 삭제
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 사용자 삭제
            cursor.execute('''
                DELETE FROM users
                WHERE id = ?
            ''', (user_id,))
            
            # 세션 삭제
            cursor.execute('''
                DELETE FROM sessions
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"사용자 삭제 중 오류 발생: {e}")
            return False 