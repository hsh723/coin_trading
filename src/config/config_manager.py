import json
import yaml
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import sqlite3
from pathlib import Path
import jsonschema
import shutil
import logging
import hashlib
import threading
import queue
import time
import pandas as pd

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self,
                 config_dir: str = "./config",
                 data_dir: str = "./data",
                 backup_dir: str = "./backup"):
        """
        설정 관리자 초기화
        
        Args:
            config_dir: 설정 디렉토리
            data_dir: 데이터 디렉토리
            backup_dir: 백업 디렉토리
        """
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        
        # 로거 설정
        self.logger = logging.getLogger("config_manager")
        
        # 설정 큐
        self.config_queue = queue.Queue()
        
        # 데이터베이스 연결
        self.db_path = os.path.join(data_dir, "config.db")
        self._init_database()
        
        # 디렉토리 생성
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        
        # 설정 스키마
        self.schema = {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "settings": {
                    "type": "object",
                    "properties": {
                        "general": {
                            "type": "object",
                            "properties": {
                                "debug": {"type": "boolean"},
                                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
                            },
                            "required": ["debug", "log_level"]
                        },
                        "database": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["sqlite", "mysql", "postgresql"]},
                                "host": {"type": "string"},
                                "port": {"type": "integer"},
                                "name": {"type": "string"},
                                "user": {"type": "string"},
                                "password": {"type": "string"}
                            },
                            "required": ["type", "name"]
                        },
                        "api": {
                            "type": "object",
                            "properties": {
                                "base_url": {"type": "string"},
                                "timeout": {"type": "integer"},
                                "retry_count": {"type": "integer"}
                            },
                            "required": ["base_url", "timeout"]
                        }
                    },
                    "required": ["general", "database", "api"]
                }
            },
            "required": ["version", "settings"]
        }
        
        # 설정 변경 추적 스레드
        self.tracking_thread = None
        self.is_running = False
        
    def _init_database(self) -> None:
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 설정 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configs (
                    name TEXT,
                    version TEXT,
                    content TEXT,
                    hash TEXT,
                    created_at DATETIME,
                    updated_at DATETIME,
                    PRIMARY KEY (name)
                )
            ''')
            
            # 설정 변경 기록 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_name TEXT,
                    old_version TEXT,
                    new_version TEXT,
                    change_type TEXT,
                    changed_at DATETIME,
                    changed_by TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 중 오류 발생: {e}")
            raise
            
    def start_tracking(self) -> None:
        """설정 변경 추적 시작"""
        try:
            self.is_running = True
            self.tracking_thread = threading.Thread(target=self._track_changes)
            self.tracking_thread.start()
            
        except Exception as e:
            self.logger.error(f"설정 변경 추적 시작 중 오류 발생: {e}")
            raise
            
    def stop_tracking(self) -> None:
        """설정 변경 추적 중지"""
        try:
            self.is_running = False
            if self.tracking_thread:
                self.tracking_thread.join()
                
        except Exception as e:
            self.logger.error(f"설정 변경 추적 중지 중 오류 발생: {e}")
            raise
            
    def _track_changes(self) -> None:
        """설정 변경 추적 루프"""
        try:
            while self.is_running:
                if not self.config_queue.empty():
                    change = self.config_queue.get()
                    self._record_change(change)
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"설정 변경 추적 중 오류 발생: {e}")
            
    def load_config(self, name: str, format: str = "json") -> Dict[str, Any]:
        """
        설정 로드
        
        Args:
            name: 설정 이름
            format: 설정 파일 형식 (json/yaml)
            
        Returns:
            설정 내용
        """
        try:
            # 설정 파일 경로
            config_path = os.path.join(self.config_dir, f"{name}.{format}")
            
            # 설정 파일 로드
            with open(config_path, 'r') as f:
                if format == "json":
                    config = json.load(f)
                elif format == "yaml":
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"지원하지 않는 형식: {format}")
                    
            # 설정 검증
            self._validate_config(config)
            
            # 설정 저장
            self._save_config(name, config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"설정 로드 중 오류 발생: {e}")
            raise
            
    def save_config(self,
                    name: str,
                    config: Dict[str, Any],
                    format: str = "json") -> None:
        """
        설정 저장
        
        Args:
            name: 설정 이름
            config: 설정 내용
            format: 설정 파일 형식 (json/yaml)
        """
        try:
            # 설정 검증
            self._validate_config(config)
            
            # 설정 파일 경로
            config_path = os.path.join(self.config_dir, f"{name}.{format}")
            
            # 설정 파일 저장
            with open(config_path, 'w') as f:
                if format == "json":
                    json.dump(config, f, indent=4)
                elif format == "yaml":
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    raise ValueError(f"지원하지 않는 형식: {format}")
                    
            # 설정 저장
            self._save_config(name, config)
            
            # 설정 변경 기록
            self.config_queue.put({
                "config_name": name,
                "change_type": "UPDATE",
                "changed_by": "system"
            })
            
        except Exception as e:
            self.logger.error(f"설정 저장 중 오류 발생: {e}")
            raise
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        설정 검증
        
        Args:
            config: 설정 내용
        """
        try:
            jsonschema.validate(instance=config, schema=self.schema)
            
        except jsonschema.exceptions.ValidationError as e:
            self.logger.error(f"설정 검증 실패: {e}")
            raise
            
    def _save_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        설정 데이터베이스에 저장
        
        Args:
            name: 설정 이름
            config: 설정 내용
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 설정 해시 계산
            config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
            
            # 현재 시간
            now = datetime.now()
            
            cursor.execute('''
                INSERT OR REPLACE INTO configs
                (name, version, content, hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM configs WHERE name = ?), ?), ?)
            ''', (
                name,
                config["version"],
                json.dumps(config),
                config_hash,
                name,
                now,
                now
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"설정 저장 중 오류 발생: {e}")
            raise
            
    def _record_change(self, change: Dict[str, Any]) -> None:
        """
        설정 변경 기록
        
        Args:
            change: 변경 정보
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 현재 설정 조회
            cursor.execute('''
                SELECT version FROM configs
                WHERE name = ?
            ''', (change["config_name"],))
            
            result = cursor.fetchone()
            old_version = result[0] if result else None
            
            # 새로운 설정 조회
            cursor.execute('''
                SELECT version FROM configs
                WHERE name = ?
            ''', (change["config_name"],))
            
            result = cursor.fetchone()
            new_version = result[0] if result else None
            
            # 변경 기록 저장
            cursor.execute('''
                INSERT INTO config_changes
                (config_name, old_version, new_version, change_type, changed_at, changed_by)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                change["config_name"],
                old_version,
                new_version,
                change["change_type"],
                datetime.now(),
                change["changed_by"]
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"설정 변경 기록 중 오류 발생: {e}")
            raise
            
    def backup_config(self, name: str) -> None:
        """
        설정 백업
        
        Args:
            name: 설정 이름
        """
        try:
            # 백업 디렉토리 생성
            backup_path = os.path.join(self.backup_dir, name)
            os.makedirs(backup_path, exist_ok=True)
            
            # 설정 파일 백업
            for format in ["json", "yaml"]:
                config_path = os.path.join(self.config_dir, f"{name}.{format}")
                if os.path.exists(config_path):
                    backup_file = os.path.join(backup_path, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}")
                    shutil.copy2(config_path, backup_file)
                    
            # 데이터베이스 백업
            db_backup = os.path.join(backup_path, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            shutil.copy2(self.db_path, db_backup)
            
        except Exception as e:
            self.logger.error(f"설정 백업 중 오류 발생: {e}")
            raise
            
    def restore_config(self, name: str, backup_time: datetime) -> None:
        """
        설정 복원
        
        Args:
            name: 설정 이름
            backup_time: 백업 시간
        """
        try:
            # 백업 디렉토리 경로
            backup_path = os.path.join(self.backup_dir, name)
            
            # 설정 파일 복원
            for format in ["json", "yaml"]:
                backup_file = os.path.join(backup_path, f"{name}_{backup_time.strftime('%Y%m%d_%H%M%S')}.{format}")
                if os.path.exists(backup_file):
                    config_path = os.path.join(self.config_dir, f"{name}.{format}")
                    shutil.copy2(backup_file, config_path)
                    
            # 데이터베이스 복원
            db_backup = os.path.join(backup_path, f"config_{backup_time.strftime('%Y%m%d_%H%M%S')}.db")
            if os.path.exists(db_backup):
                shutil.copy2(db_backup, self.db_path)
                
        except Exception as e:
            self.logger.error(f"설정 복원 중 오류 발생: {e}")
            raise
            
    def get_config_history(self, name: str) -> pd.DataFrame:
        """
        설정 변경 이력 조회
        
        Args:
            name: 설정 이름
            
        Returns:
            변경 이력 데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            history = pd.read_sql_query('''
                SELECT * FROM config_changes
                WHERE config_name = ?
                ORDER BY changed_at DESC
            ''', conn, params=(name,))
            
            conn.close()
            
            return history
            
        except Exception as e:
            self.logger.error(f"설정 변경 이력 조회 중 오류 발생: {e}")
            raise
            
    def get_config_versions(self, name: str) -> pd.DataFrame:
        """
        설정 버전 조회
        
        Args:
            name: 설정 이름
            
        Returns:
            버전 데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            versions = pd.read_sql_query('''
                SELECT version, created_at, updated_at FROM configs
                WHERE name = ?
                ORDER BY updated_at DESC
            ''', conn, params=(name,))
            
            conn.close()
            
            return versions
            
        except Exception as e:
            self.logger.error(f"설정 버전 조회 중 오류 발생: {e}")
            raise 