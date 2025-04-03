"""
데이터베이스 관리자 모듈
"""

import os
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional
import json

from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    데이터베이스 관리자 클래스
    """
    def __init__(self):
        self.config = get_config()
        self.db_path = self.config['database']['path']
        self.backup_dir = self.config['database']['backup_dir']
        self.connection = None
        self.cursor = None
        
        # 데이터베이스 디렉토리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # 데이터베이스 초기화
        self.initialize_database()
        
    def initialize_database(self):
        """
        데이터베이스 초기화
        """
        try:
            # 기존 데이터베이스 파일이 있다면 삭제
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            # 새로운 데이터베이스 연결
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            
            # 테이블 생성
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    volume REAL NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    profit_loss REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades (id)
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL
                )
            """)
            
            self.connection.commit()
            logger.info("데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            raise
            
    async def check_connection(self) -> bool:
        """
        데이터베이스 연결 확인
        
        Returns:
            bool: 연결 상태
        """
        try:
            if self.connection is None:
                self.initialize_database()
            
            self.cursor.execute("SELECT 1")
            return True
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 확인 실패: {str(e)}")
            return False
            
    async def save_trade_data(self, trade_data: Dict[str, Any]):
        """
        거래 데이터 저장
        
        Args:
            trade_data (Dict[str, Any]): 거래 데이터
        """
        try:
            self.cursor.execute("""
                INSERT INTO trades (
                    id, symbol, entry_time, exit_time, 
                    entry_price, exit_price, volume, 
                    side, status, profit_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['id'],
                trade_data['symbol'],
                trade_data['entry_time'],
                trade_data['exit_time'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['volume'],
                trade_data['side'],
                trade_data['status'],
                trade_data['profit_loss']
            ))
            
            self.connection.commit()
            logger.info(f"거래 데이터 저장 완료: {trade_data['id']}")
            
        except Exception as e:
            logger.error(f"거래 데이터 저장 실패: {str(e)}")
            raise
            
    async def save_analysis_result(self, analysis_data: Dict[str, Any]):
        """
        분석 결과 저장
        
        Args:
            analysis_data (Dict[str, Any]): 분석 결과 데이터
        """
        try:
            self.cursor.execute("""
                INSERT INTO analysis_results (
                    id, trade_id, analysis_type, results
                ) VALUES (?, ?, ?, ?)
            """, (
                analysis_data['id'],
                analysis_data['trade_id'],
                analysis_data['analysis_type'],
                json.dumps(analysis_data['results'])
            ))
            
            self.connection.commit()
            logger.info(f"분석 결과 저장 완료: {analysis_data['id']}")
            
        except Exception as e:
            logger.error(f"분석 결과 저장 실패: {str(e)}")
            raise
            
    async def send_notification(self, notification_type: str, message: str) -> bool:
        """
        알림 전송 및 저장
        
        Args:
            notification_type (str): 알림 유형
            message (str): 알림 메시지
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            notification_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            self.cursor.execute("""
                INSERT INTO notifications (
                    id, type, message, status
                ) VALUES (?, ?, ?, ?)
            """, (
                notification_id,
                notification_type,
                message,
                'sent'
            ))
            
            self.connection.commit()
            logger.info(f"알림 저장 완료: {notification_id}")
            return True
            
        except Exception as e:
            logger.error(f"알림 저장 실패: {str(e)}")
            return False
            
    def close(self):
        """
        데이터베이스 연결 종료
        """
        try:
            if self.connection:
                self.connection.close()
                logger.info("데이터베이스 연결 종료")
                
        except Exception as e:
            logger.error(f"데이터베이스 연결 종료 실패: {str(e)}")

# 전역 인스턴스 생성
database_manager = DatabaseManager() 