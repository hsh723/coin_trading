import os
import json
import secrets
from typing import Dict, Optional
from datetime import datetime, timedelta
from ..utils.logger import setup_logger
from ..database.database import Database

class SessionManager:
    """세션 관리 클래스"""
    
    def __init__(self):
        """세션 관리 클래스 초기화"""
        self.logger = setup_logger('session_manager')
        self.db = Database()
        self.sessions = {}
        self.session_timeout = timedelta(hours=1)  # 세션 타임아웃: 1시간
        
    def create_session(self, user_id: str, user_info: Dict) -> str:
        """
        세션 생성
        
        Args:
            user_id (str): 사용자 ID
            user_info (Dict): 사용자 정보
            
        Returns:
            str: 세션 토큰
        """
        try:
            # 세션 토큰 생성
            session_token = secrets.token_urlsafe(32)
            
            # 세션 정보 설정
            session_data = {
                'user_id': user_id,
                'user_info': user_info,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'ip_address': user_info.get('ip_address', ''),
                'user_agent': user_info.get('user_agent', '')
            }
            
            # 세션 저장
            self.sessions[session_token] = session_data
            self.db.save_session(session_token, session_data)
            
            self.logger.info(f"세션 생성 완료: {user_id}")
            return session_token
            
        except Exception as e:
            self.logger.error(f"세션 생성 실패: {str(e)}")
            raise
            
    def validate_session(self, session_token: str) -> bool:
        """
        세션 유효성 검사
        
        Args:
            session_token (str): 세션 토큰
            
        Returns:
            bool: 세션 유효성 여부
        """
        try:
            # 세션 조회
            session_data = self.sessions.get(session_token)
            if not session_data:
                session_data = self.db.get_session(session_token)
                if not session_data:
                    return False
                    
            # 세션 타임아웃 확인
            if datetime.now() - session_data['last_activity'] > self.session_timeout:
                self.delete_session(session_token)
                return False
                
            # 마지막 활동 시간 업데이트
            session_data['last_activity'] = datetime.now()
            self.sessions[session_token] = session_data
            self.db.update_session(session_token, session_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"세션 유효성 검사 실패: {str(e)}")
            return False
            
    def get_session(self, session_token: str) -> Optional[Dict]:
        """
        세션 정보 조회
        
        Args:
            session_token (str): 세션 토큰
            
        Returns:
            Optional[Dict]: 세션 정보
        """
        try:
            if self.validate_session(session_token):
                return self.sessions.get(session_token)
            return None
            
        except Exception as e:
            self.logger.error(f"세션 정보 조회 실패: {str(e)}")
            return None
            
    def delete_session(self, session_token: str) -> None:
        """
        세션 삭제
        
        Args:
            session_token (str): 세션 토큰
        """
        try:
            if session_token in self.sessions:
                del self.sessions[session_token]
            self.db.delete_session(session_token)
            self.logger.info(f"세션 삭제 완료: {session_token}")
            
        except Exception as e:
            self.logger.error(f"세션 삭제 실패: {str(e)}")
            raise
            
    def cleanup_expired_sessions(self) -> None:
        """만료된 세션 정리"""
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            # 메모리에서 만료된 세션 찾기
            for token, session in self.sessions.items():
                if current_time - session['last_activity'] > self.session_timeout:
                    expired_sessions.append(token)
                    
            # 만료된 세션 삭제
            for token in expired_sessions:
                self.delete_session(token)
                
            # 데이터베이스에서 만료된 세션 삭제
            self.db.cleanup_expired_sessions(self.session_timeout)
            
            self.logger.info(f"만료된 세션 {len(expired_sessions)}개 정리 완료")
            
        except Exception as e:
            self.logger.error(f"만료된 세션 정리 실패: {str(e)}")
            raise
            
    def update_session_info(self, session_token: str, info: Dict) -> None:
        """
        세션 정보 업데이트
        
        Args:
            session_token (str): 세션 토큰
            info (Dict): 업데이트할 정보
        """
        try:
            if session_token in self.sessions:
                self.sessions[session_token].update(info)
                self.sessions[session_token]['last_activity'] = datetime.now()
                self.db.update_session(session_token, self.sessions[session_token])
                
        except Exception as e:
            self.logger.error(f"세션 정보 업데이트 실패: {str(e)}")
            raise 