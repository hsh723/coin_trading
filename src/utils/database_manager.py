"""
데이터베이스 관리 모듈
데이터베이스 연결, 쿼리 실행, 트랜잭션 관리 등의 기능을 제공합니다.
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = "data/trading.db"):
        """
        데이터베이스 관리자 초기화
        
        Args:
            db_path (str): 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self._setup_logging()
        self._ensure_db_directory()
        
    def _setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger(__name__)
        
    def _ensure_db_directory(self):
        """데이터베이스 디렉토리 생성"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            self.logger.info(f"데이터베이스 연결 성공: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"데이터베이스 연결 실패: {e}")
            raise
            
    def disconnect(self):
        """데이터베이스 연결 종료"""
        if self.connection:
            self.connection.close()
            self.logger.info("데이터베이스 연결 종료")
            
    def execute(self, query: str, params: tuple = None) -> None:
        """
        SQL 쿼리 실행
        
        Args:
            query (str): 실행할 SQL 쿼리
            params (tuple): 쿼리 파라미터
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.connection.commit()
        except sqlite3.Error as e:
            self.logger.error(f"쿼리 실행 실패: {e}")
            raise
            
    def fetch_one(self, query: str, params: tuple = None) -> Optional[tuple]:
        """
        단일 레코드 조회
        
        Args:
            query (str): 실행할 SQL 쿼리
            params (tuple): 쿼리 파라미터
            
        Returns:
            Optional[tuple]: 조회된 레코드
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            self.logger.error(f"레코드 조회 실패: {e}")
            raise
            
    def fetch_all(self, query: str, params: tuple = None) -> List[tuple]:
        """
        모든 레코드 조회
        
        Args:
            query (str): 실행할 SQL 쿼리
            params (tuple): 쿼리 파라미터
            
        Returns:
            List[tuple]: 조회된 레코드 목록
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"레코드 조회 실패: {e}")
            raise
            
    def create_table(self, table_name: str, columns: Dict[str, str]) -> None:
        """
        테이블 생성
        
        Args:
            table_name (str): 테이블 이름
            columns (Dict[str, str]): 컬럼 정의 (컬럼명: 데이터타입)
        """
        try:
            columns_str = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
            self.execute(query)
            self.logger.info(f"테이블 생성 성공: {table_name}")
        except sqlite3.Error as e:
            self.logger.error(f"테이블 생성 실패: {e}")
            raise
            
    def insert(self, table_name: str, data: Dict[str, Any]) -> None:
        """
        레코드 삽입
        
        Args:
            table_name (str): 테이블 이름
            data (Dict[str, Any]): 삽입할 데이터
        """
        try:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            self.execute(query, tuple(data.values()))
            self.logger.info(f"데이터 삽입 성공: {table_name}")
        except sqlite3.Error as e:
            self.logger.error(f"데이터 삽입 실패: {e}")
            raise
            
    def update(self, table_name: str, data: Dict[str, Any], where: str, params: tuple) -> None:
        """
        레코드 업데이트
        
        Args:
            table_name (str): 테이블 이름
            data (Dict[str, Any]): 업데이트할 데이터
            where (str): WHERE 절
            params (tuple): WHERE 절 파라미터
        """
        try:
            set_clause = ", ".join([f"{col} = ?" for col in data.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where}"
            self.execute(query, tuple(data.values()) + params)
            self.logger.info(f"데이터 업데이트 성공: {table_name}")
        except sqlite3.Error as e:
            self.logger.error(f"데이터 업데이트 실패: {e}")
            raise
            
    def delete(self, table_name: str, where: str, params: tuple) -> None:
        """
        레코드 삭제
        
        Args:
            table_name (str): 테이블 이름
            where (str): WHERE 절
            params (tuple): WHERE 절 파라미터
        """
        try:
            query = f"DELETE FROM {table_name} WHERE {where}"
            self.execute(query, params)
            self.logger.info(f"데이터 삭제 성공: {table_name}")
        except sqlite3.Error as e:
            self.logger.error(f"데이터 삭제 실패: {e}")
            raise
            
    def begin_transaction(self) -> None:
        """트랜잭션 시작"""
        try:
            self.execute("BEGIN TRANSACTION")
            self.logger.info("트랜잭션 시작")
        except sqlite3.Error as e:
            self.logger.error(f"트랜잭션 시작 실패: {e}")
            raise
            
    def commit_transaction(self) -> None:
        """트랜잭션 커밋"""
        try:
            self.connection.commit()
            self.logger.info("트랜잭션 커밋")
        except sqlite3.Error as e:
            self.logger.error(f"트랜잭션 커밋 실패: {e}")
            raise
            
    def rollback_transaction(self) -> None:
        """트랜잭션 롤백"""
        try:
            self.connection.rollback()
            self.logger.info("트랜잭션 롤백")
        except sqlite3.Error as e:
            self.logger.error(f"트랜잭션 롤백 실패: {e}")
            raise 