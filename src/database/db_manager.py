import os
import json
import logging
import threading
import queue
import time
import sqlite3
import psycopg2
import mysql.connector
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    """데이터베이스 클래스"""
    
    def __init__(self,
                 config_dir: str = "./config",
                 data_dir: str = "./data"):
        """
        데이터베이스 초기화
        
        Args:
            config_dir: 설정 디렉토리
            data_dir: 데이터 디렉토리
        """
        self.config_dir = config_dir
        self.data_dir = data_dir
        
        # 로거 설정
        self.logger = logging.getLogger("database")
        
        # 쿼리 큐
        self.query_queue = queue.Queue()
        
        # 디렉토리 생성
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 설정 로드
        self.config = self._load_config()
        
        # 데이터베이스 연결
        self.connections: Dict[str, Any] = {}
        
        # 데이터베이스 관리자
        self.is_running = False
        
        # 트랜잭션 관리
        self.transactions: Dict[str, Dict[str, Any]] = {}
        
        # 통계
        self.stats = {
            "connections": 0,
            "queries_executed": 0,
            "transactions": 0,
            "errors": 0
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            config_path = os.path.join(self.config_dir, "db_config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {
                "default": {
                    "type": "sqlite",
                    "database": "database.db"
                }
            }
            
    def start(self) -> None:
        """데이터베이스 시작"""
        try:
            self.is_running = True
            
            # 쿼리 처리 시작
            threading.Thread(target=self._process_queries, daemon=True).start()
            
            self.logger.info("데이터베이스가 시작되었습니다")
            
        except Exception as e:
            self.logger.error(f"데이터베이스 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """데이터베이스 중지"""
        try:
            self.is_running = False
            
            # 연결 종료
            for connection in self.connections.values():
                self._close_connection(connection)
                
            self.logger.info("데이터베이스가 중지되었습니다")
            
        except Exception as e:
            self.logger.error(f"데이터베이스 중지 중 오류 발생: {e}")
            raise
            
    def _create_connection(self,
                         db_type: str,
                         config: Dict[str, Any]) -> Any:
        """
        데이터베이스 연결 생성
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            
        Returns:
            데이터베이스 연결
        """
        try:
            if db_type == "sqlite":
                # SQLite 연결
                return sqlite3.connect(
                    os.path.join(self.data_dir, config["database"]),
                    check_same_thread=False
                )
                
            elif db_type == "postgresql":
                # PostgreSQL 연결
                return psycopg2.connect(
                    host=config["host"],
                    port=config["port"],
                    database=config["database"],
                    user=config["user"],
                    password=config["password"]
                )
                
            elif db_type == "mysql":
                # MySQL 연결
                return mysql.connector.connect(
                    host=config["host"],
                    port=config["port"],
                    database=config["database"],
                    user=config["user"],
                    password=config["password"]
                )
                
            else:
                raise ValueError(f"지원하지 않는 데이터베이스 타입: {db_type}")
                
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 생성 중 오류 발생: {e}")
            raise
            
    def _close_connection(self, connection: Any) -> None:
        """
        데이터베이스 연결 종료
        
        Args:
            connection: 데이터베이스 연결
        """
        try:
            if isinstance(connection, sqlite3.Connection):
                connection.close()
            elif isinstance(connection, psycopg2.extensions.connection):
                connection.close()
            elif isinstance(connection, mysql.connector.connection.MySQLConnection):
                connection.close()
                
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 종료 중 오류 발생: {e}")
            
    def get_connection(self,
                      db_type: str,
                      config: Dict[str, Any]) -> Any:
        """
        데이터베이스 연결 가져오기
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            
        Returns:
            데이터베이스 연결
        """
        try:
            # 연결 키 생성
            connection_key = f"{db_type}_{json.dumps(config, sort_keys=True)}"
            
            # 연결 확인
            if connection_key not in self.connections:
                self.connections[connection_key] = self._create_connection(db_type, config)
                self.stats["connections"] += 1
                
            return self.connections[connection_key]
            
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 가져오기 중 오류 발생: {e}")
            raise
            
    def _process_queries(self) -> None:
        """쿼리 처리 루프"""
        try:
            while self.is_running:
                if not self.query_queue.empty():
                    query = self.query_queue.get()
                    self._handle_query(
                        query["db_type"],
                        query["config"],
                        query["operation"],
                        query["sql"],
                        query.get("params"),
                        query.get("transaction_id")
                    )
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"쿼리 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def _handle_query(self,
                     db_type: str,
                     config: Dict[str, Any],
                     operation: str,
                     sql: str,
                     params: Optional[Tuple] = None,
                     transaction_id: Optional[str] = None) -> None:
        """
        쿼리 작업 처리
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            operation: 작업 타입
            sql: SQL 쿼리
            params: 쿼리 파라미터
            transaction_id: 트랜잭션 ID
        """
        try:
            if operation == "execute":
                # 쿼리 실행
                self._execute_query(db_type, config, sql, params, transaction_id)
                
            elif operation == "fetch":
                # 쿼리 결과 조회
                return self._fetch_query(db_type, config, sql, params, transaction_id)
                
            elif operation == "begin":
                # 트랜잭션 시작
                self._begin_transaction(db_type, config, transaction_id)
                
            elif operation == "commit":
                # 트랜잭션 커밋
                self._commit_transaction(db_type, config, transaction_id)
                
            elif operation == "rollback":
                # 트랜잭션 롤백
                self._rollback_transaction(db_type, config, transaction_id)
                
        except Exception as e:
            self.logger.error(f"쿼리 작업 처리 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _execute_query(self,
                      db_type: str,
                      config: Dict[str, Any],
                      sql: str,
                      params: Optional[Tuple] = None,
                      transaction_id: Optional[str] = None) -> None:
        """
        쿼리 실행
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            sql: SQL 쿼리
            params: 쿼리 파라미터
            transaction_id: 트랜잭션 ID
        """
        try:
            connection = self.get_connection(db_type, config)
            cursor = connection.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
                
            connection.commit()
            self.stats["queries_executed"] += 1
            
        except Exception as e:
            self.logger.error(f"쿼리 실행 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _fetch_query(self,
                    db_type: str,
                    config: Dict[str, Any],
                    sql: str,
                    params: Optional[Tuple] = None,
                    transaction_id: Optional[str] = None) -> List[Tuple]:
        """
        쿼리 결과 조회
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            sql: SQL 쿼리
            params: 쿼리 파라미터
            transaction_id: 트랜잭션 ID
            
        Returns:
            쿼리 결과
        """
        try:
            connection = self.get_connection(db_type, config)
            cursor = connection.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
                
            results = cursor.fetchall()
            self.stats["queries_executed"] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"쿼리 결과 조회 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _begin_transaction(self,
                         db_type: str,
                         config: Dict[str, Any],
                         transaction_id: str) -> None:
        """
        트랜잭션 시작
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            transaction_id: 트랜잭션 ID
        """
        try:
            connection = self.get_connection(db_type, config)
            cursor = connection.cursor()
            
            cursor.execute("BEGIN")
            
            # 트랜잭션 관리
            self.transactions[transaction_id] = {
                "db_type": db_type,
                "config": config,
                "connection": connection,
                "cursor": cursor
            }
            
            self.stats["transactions"] += 1
            
        except Exception as e:
            self.logger.error(f"트랜잭션 시작 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _commit_transaction(self,
                          db_type: str,
                          config: Dict[str, Any],
                          transaction_id: str) -> None:
        """
        트랜잭션 커밋
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            transaction_id: 트랜잭션 ID
        """
        try:
            if transaction_id in self.transactions:
                transaction = self.transactions[transaction_id]
                
                transaction["cursor"].execute("COMMIT")
                transaction["connection"].commit()
                
                del self.transactions[transaction_id]
                
        except Exception as e:
            self.logger.error(f"트랜잭션 커밋 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def _rollback_transaction(self,
                            db_type: str,
                            config: Dict[str, Any],
                            transaction_id: str) -> None:
        """
        트랜잭션 롤백
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            transaction_id: 트랜잭션 ID
        """
        try:
            if transaction_id in self.transactions:
                transaction = self.transactions[transaction_id]
                
                transaction["cursor"].execute("ROLLBACK")
                transaction["connection"].rollback()
                
                del self.transactions[transaction_id]
                
        except Exception as e:
            self.logger.error(f"트랜잭션 롤백 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def execute(self,
               db_type: str,
               config: Dict[str, Any],
               sql: str,
               params: Optional[Tuple] = None,
               transaction_id: Optional[str] = None) -> None:
        """
        쿼리 실행 요청
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            sql: SQL 쿼리
            params: 쿼리 파라미터
            transaction_id: 트랜잭션 ID
        """
        try:
            self.query_queue.put({
                "db_type": db_type,
                "config": config,
                "operation": "execute",
                "sql": sql,
                "params": params,
                "transaction_id": transaction_id
            })
            
        except Exception as e:
            self.logger.error(f"쿼리 실행 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def fetch(self,
             db_type: str,
             config: Dict[str, Any],
             sql: str,
             params: Optional[Tuple] = None,
             transaction_id: Optional[str] = None) -> List[Tuple]:
        """
        쿼리 결과 조회 요청
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            sql: SQL 쿼리
            params: 쿼리 파라미터
            transaction_id: 트랜잭션 ID
            
        Returns:
            쿼리 결과
        """
        try:
            result_queue = queue.Queue()
            
            self.query_queue.put({
                "db_type": db_type,
                "config": config,
                "operation": "fetch",
                "sql": sql,
                "params": params,
                "transaction_id": transaction_id,
                "result_queue": result_queue
            })
            
            return result_queue.get()
            
        except Exception as e:
            self.logger.error(f"쿼리 결과 조회 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            raise
            
    def begin_transaction(self,
                         db_type: str,
                         config: Dict[str, Any],
                         transaction_id: str) -> None:
        """
        트랜잭션 시작 요청
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            transaction_id: 트랜잭션 ID
        """
        try:
            self.query_queue.put({
                "db_type": db_type,
                "config": config,
                "operation": "begin",
                "transaction_id": transaction_id
            })
            
        except Exception as e:
            self.logger.error(f"트랜잭션 시작 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def commit_transaction(self,
                          db_type: str,
                          config: Dict[str, Any],
                          transaction_id: str) -> None:
        """
        트랜잭션 커밋 요청
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            transaction_id: 트랜잭션 ID
        """
        try:
            self.query_queue.put({
                "db_type": db_type,
                "config": config,
                "operation": "commit",
                "transaction_id": transaction_id
            })
            
        except Exception as e:
            self.logger.error(f"트랜잭션 커밋 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def rollback_transaction(self,
                            db_type: str,
                            config: Dict[str, Any],
                            transaction_id: str) -> None:
        """
        트랜잭션 롤백 요청
        
        Args:
            db_type: 데이터베이스 타입
            config: 연결 설정
            transaction_id: 트랜잭션 ID
        """
        try:
            self.query_queue.put({
                "db_type": db_type,
                "config": config,
                "operation": "rollback",
                "transaction_id": transaction_id
            })
            
        except Exception as e:
            self.logger.error(f"트랜잭션 롤백 요청 중 오류 발생: {e}")
            self.stats["errors"] += 1
            
    def get_connection_count(self) -> int:
        """
        연결 수 조회
        
        Returns:
            연결 수
        """
        return self.stats["connections"]
        
    def get_queries_executed_count(self) -> int:
        """
        실행된 쿼리 수 조회
        
        Returns:
            실행된 쿼리 수
        """
        return self.stats["queries_executed"]
        
    def get_transaction_count(self) -> int:
        """
        트랜잭션 수 조회
        
        Returns:
            트랜잭션 수
        """
        return self.stats["transactions"]
        
    def get_stats(self) -> Dict[str, int]:
        """
        데이터베이스 통계 조회
        
        Returns:
            데이터베이스 통계
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """데이터베이스 통계 초기화"""
        self.stats = {
            "connections": 0,
            "queries_executed": 0,
            "transactions": 0,
            "errors": 0
        } 