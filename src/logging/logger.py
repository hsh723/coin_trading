import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
from pathlib import Path
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import threading
import queue
import gzip
import shutil
import time

class Logger:
    """로깅 시스템 클래스"""
    
    def __init__(self,
                 config_path: str = "./config/logger_config.json",
                 log_dir: str = "./logs",
                 data_dir: str = "./data"):
        """
        로거 초기화
        
        Args:
            config_path: 설정 파일 경로
            log_dir: 로그 디렉토리
            data_dir: 데이터 디렉토리
        """
        self.config_path = config_path
        self.log_dir = log_dir
        self.data_dir = data_dir
        
        # 설정 로드
        self.config = self._load_config()
        
        # 로그 큐
        self.log_queue = queue.Queue()
        
        # 데이터베이스 연결
        self.db_path = os.path.join(data_dir, "logging.db")
        self._init_database()
        
        # 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # 로깅 스레드
        self.logging_thread = None
        self.is_running = False
        
        # 로그 패턴
        self.log_patterns = {
            "error": re.compile(r"ERROR|Exception|Traceback"),
            "warning": re.compile(r"WARNING"),
            "info": re.compile(r"INFO"),
            "debug": re.compile(r"DEBUG")
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def _init_database(self) -> None:
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 로그 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    timestamp DATETIME,
                    level TEXT,
                    module TEXT,
                    message TEXT,
                    stack_trace TEXT,
                    PRIMARY KEY (timestamp, level, module, message)
                )
            ''')
            
            # 로그 패턴 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS log_patterns (
                    pattern TEXT,
                    count INTEGER,
                    last_seen DATETIME,
                    PRIMARY KEY (pattern)
                )
            ''')
            
            # 로그 통계 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS log_stats (
                    date DATE,
                    level TEXT,
                    count INTEGER,
                    PRIMARY KEY (date, level)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"데이터베이스 초기화 중 오류 발생: {e}")
            raise
            
    def start(self) -> None:
        """로깅 시작"""
        try:
            self.is_running = True
            self.logging_thread = threading.Thread(target=self._process_logs)
            self.logging_thread.start()
            
        except Exception as e:
            print(f"로깅 시작 중 오류 발생: {e}")
            raise
            
    def stop(self) -> None:
        """로깅 중지"""
        try:
            self.is_running = False
            if self.logging_thread:
                self.logging_thread.join()
                
        except Exception as e:
            print(f"로깅 중지 중 오류 발생: {e}")
            raise
            
    def _process_logs(self) -> None:
        """로그 처리 루프"""
        try:
            while self.is_running:
                if not self.log_queue.empty():
                    log_entry = self.log_queue.get()
                    self._save_log(log_entry)
                    self._analyze_log(log_entry)
                    
                time.sleep(0.1)
                
        except Exception as e:
            print(f"로그 처리 중 오류 발생: {e}")
            
    def log(self,
            level: str,
            module: str,
            message: str,
            stack_trace: Optional[str] = None) -> None:
        """
        로그 기록
        
        Args:
            level: 로그 레벨
            module: 모듈 이름
            message: 로그 메시지
            stack_trace: 스택 트레이스 (선택사항)
        """
        try:
            log_entry = {
                "timestamp": datetime.now(),
                "level": level,
                "module": module,
                "message": message,
                "stack_trace": stack_trace
            }
            
            # 로그 큐에 추가
            self.log_queue.put(log_entry)
            
            # 로그 파일에 기록
            self._write_to_file(log_entry)
            
        except Exception as e:
            print(f"로그 기록 중 오류 발생: {e}")
            
    def _write_to_file(self, log_entry: Dict[str, Any]) -> None:
        """
        로그 파일에 기록
        
        Args:
            log_entry: 로그 항목
        """
        try:
            # 로그 파일 경로
            date_str = log_entry["timestamp"].strftime("%Y-%m-%d")
            log_file = os.path.join(self.log_dir, f"{date_str}.log")
            
            # 로그 포맷
            log_format = self.config.get("log_format", "%(asctime)s - %(levelname)s - %(module)s - %(message)s")
            log_message = log_format % {
                "asctime": log_entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "levelname": log_entry["level"],
                "module": log_entry["module"],
                "message": log_entry["message"]
            }
            
            # 로그 파일에 기록
            with open(log_file, "a") as f:
                f.write(log_message + "\n")
                
            # 스택 트레이스 기록
            if log_entry["stack_trace"]:
                with open(log_file, "a") as f:
                    f.write(log_entry["stack_trace"] + "\n")
                    
        except Exception as e:
            print(f"로그 파일 기록 중 오류 발생: {e}")
            
    def _save_log(self, log_entry: Dict[str, Any]) -> None:
        """
        로그 데이터베이스에 저장
        
        Args:
            log_entry: 로그 항목
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO logs
                (timestamp, level, module, message, stack_trace)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                log_entry["timestamp"],
                log_entry["level"],
                log_entry["module"],
                log_entry["message"],
                log_entry["stack_trace"]
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"로그 저장 중 오류 발생: {e}")
            
    def _analyze_log(self, log_entry: Dict[str, Any]) -> None:
        """
        로그 분석
        
        Args:
            log_entry: 로그 항목
        """
        try:
            # 로그 패턴 분석
            for pattern_name, pattern in self.log_patterns.items():
                if pattern.search(log_entry["message"]):
                    self._update_pattern_stats(pattern_name, log_entry["message"])
                    
            # 로그 통계 업데이트
            self._update_log_stats(log_entry)
            
        except Exception as e:
            print(f"로그 분석 중 오류 발생: {e}")
            
    def _update_pattern_stats(self, pattern: str, message: str) -> None:
        """
        로그 패턴 통계 업데이트
        
        Args:
            pattern: 패턴 이름
            message: 로그 메시지
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO log_patterns
                (pattern, count, last_seen)
                VALUES (?, COALESCE((SELECT count + 1 FROM log_patterns WHERE pattern = ?), 1), ?)
            ''', (pattern, pattern, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"패턴 통계 업데이트 중 오류 발생: {e}")
            
    def _update_log_stats(self, log_entry: Dict[str, Any]) -> None:
        """
        로그 통계 업데이트
        
        Args:
            log_entry: 로그 항목
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            date_str = log_entry["timestamp"].strftime("%Y-%m-%d")
            
            cursor.execute('''
                INSERT OR REPLACE INTO log_stats
                (date, level, count)
                VALUES (?, ?, COALESCE((SELECT count + 1 FROM log_stats WHERE date = ? AND level = ?), 1))
            ''', (date_str, log_entry["level"], date_str, log_entry["level"]))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"로그 통계 업데이트 중 오류 발생: {e}")
            
    def get_logs(self,
                 start_time: datetime,
                 end_time: datetime,
                 level: Optional[str] = None,
                 module: Optional[str] = None) -> pd.DataFrame:
        """
        로그 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            level: 로그 레벨 (선택사항)
            module: 모듈 이름 (선택사항)
            
        Returns:
            로그 데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM logs
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_time, end_time]
            
            if level:
                query += " AND level = ?"
                params.append(level)
                
            if module:
                query += " AND module = ?"
                params.append(module)
                
            query += " ORDER BY timestamp DESC"
            
            logs = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return logs
            
        except Exception as e:
            print(f"로그 조회 중 오류 발생: {e}")
            raise
            
    def get_pattern_stats(self) -> pd.DataFrame:
        """
        로그 패턴 통계 조회
        
        Returns:
            패턴 통계 데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            stats = pd.read_sql_query('''
                SELECT * FROM log_patterns
                ORDER BY count DESC
            ''', conn)
            
            conn.close()
            
            return stats
            
        except Exception as e:
            print(f"패턴 통계 조회 중 오류 발생: {e}")
            raise
            
    def get_log_stats(self,
                      start_date: datetime,
                      end_date: datetime) -> pd.DataFrame:
        """
        로그 통계 조회
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            로그 통계 데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            stats = pd.read_sql_query('''
                SELECT * FROM log_stats
                WHERE date BETWEEN ? AND ?
                ORDER BY date, level
            ''', conn, params=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
            
            conn.close()
            
            return stats
            
        except Exception as e:
            print(f"로그 통계 조회 중 오류 발생: {e}")
            raise
            
    def compress_logs(self, days: int = 30) -> None:
        """
        오래된 로그 파일 압축
        
        Args:
            days: 압축할 로그 파일의 일수
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for log_file in os.listdir(self.log_dir):
                if log_file.endswith(".log"):
                    file_date = datetime.strptime(log_file.split(".")[0], "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        # 로그 파일 압축
                        with open(os.path.join(self.log_dir, log_file), 'rb') as f_in:
                            with gzip.open(os.path.join(self.log_dir, f"{log_file}.gz"), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                                
                        # 원본 파일 삭제
                        os.remove(os.path.join(self.log_dir, log_file))
                        
        except Exception as e:
            print(f"로그 압축 중 오류 발생: {e}")
            
    def cleanup_logs(self, days: int = 90) -> None:
        """
        오래된 로그 파일 삭제
        
        Args:
            days: 삭제할 로그 파일의 일수
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for log_file in os.listdir(self.log_dir):
                if log_file.endswith(".gz"):
                    file_date = datetime.strptime(log_file.split(".")[0], "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        os.remove(os.path.join(self.log_dir, log_file))
                        
        except Exception as e:
            print(f"로그 정리 중 오류 발생: {e}") 