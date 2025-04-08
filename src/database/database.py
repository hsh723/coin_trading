"""
데이터베이스 관리 모듈
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os
import pandas as pd
from pathlib import Path

class DatabaseManager:
    """데이터베이스 관리자 클래스"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        초기화
        
        Args:
            db_path (str): 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()
        
    def _init_db(self):
        """데이터베이스 초기화"""
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 거래 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        size REAL NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        pnl REAL,
                        status TEXT NOT NULL,
                        strategy TEXT,
                        reason TEXT
                    )
                """)
                
                # 성과 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        total_trades INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        total_pnl REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL,
                        win_rate REAL NOT NULL
                    )
                """)
                
                # 시스템 로그 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        level TEXT NOT NULL,
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        context TEXT
                    )
                """)
                
                # 알림 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        status TEXT NOT NULL,
                        channel TEXT NOT NULL
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            raise
            
    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        거래 기록 저장
        
        Args:
            trade_data (Dict[str, Any]): 거래 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, side, entry_price, exit_price, size,
                        entry_time, exit_time, pnl, status, strategy, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['entry_price'],
                    trade_data.get('exit_price'),
                    trade_data['size'],
                    trade_data['entry_time'],
                    trade_data.get('exit_time'),
                    trade_data.get('pnl'),
                    trade_data['status'],
                    trade_data.get('strategy'),
                    trade_data.get('reason')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"거래 기록 저장 실패: {str(e)}")
            return False
            
    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """
        거래 기록 업데이트
        
        Args:
            trade_id (int): 거래 ID
            update_data (Dict[str, Any]): 업데이트할 데이터
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
                values = list(update_data.values())
                values.append(trade_id)
                
                cursor.execute(f"""
                    UPDATE trades SET {set_clause} WHERE id = ?
                """, values)
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"거래 기록 업데이트 실패: {str(e)}")
            return False
            
    def get_trades(self, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  symbol: Optional[str] = None,
                  status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        거래 기록 조회
        
        Args:
            start_date (Optional[datetime]): 시작 날짜
            end_date (Optional[datetime]): 종료 날짜
            symbol (Optional[str]): 심볼
            status (Optional[str]): 상태
            
        Returns:
            List[Dict[str, Any]]: 거래 기록 목록
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND entry_time >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND entry_time <= ?"
                    params.append(end_date)
                    
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                    
                if status:
                    query += " AND status = ?"
                    params.append(status)
                    
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"거래 기록 조회 실패: {str(e)}")
            return []
            
    def save_performance(self, performance_data: Dict[str, Any]) -> bool:
        """
        성과 데이터 저장
        
        Args:
            performance_data (Dict[str, Any]): 성과 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO performance (
                        date, total_trades, winning_trades, total_pnl,
                        max_drawdown, sharpe_ratio, win_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    performance_data['date'],
                    performance_data['total_trades'],
                    performance_data['winning_trades'],
                    performance_data['total_pnl'],
                    performance_data['max_drawdown'],
                    performance_data.get('sharpe_ratio'),
                    performance_data['win_rate']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"성과 데이터 저장 실패: {str(e)}")
            return False
            
    def get_performance(self,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        성과 데이터 조회
        
        Args:
            start_date (Optional[datetime]): 시작 날짜
            end_date (Optional[datetime]): 종료 날짜
            
        Returns:
            List[Dict[str, Any]]: 성과 데이터 목록
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM performance WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                    
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"성과 데이터 조회 실패: {str(e)}")
            return []
            
    def save_system_log(self, log_data: Dict[str, Any]) -> bool:
        """
        시스템 로그 저장
        
        Args:
            log_data (Dict[str, Any]): 로그 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO system_logs (
                        timestamp, level, type, message, context
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    log_data['timestamp'],
                    log_data['level'],
                    log_data['type'],
                    log_data['message'],
                    json.dumps(log_data.get('context', {}))
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"시스템 로그 저장 실패: {str(e)}")
            return False
            
    def get_system_logs(self,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       level: Optional[str] = None,
                       log_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        시스템 로그 조회
        
        Args:
            start_date (Optional[datetime]): 시작 날짜
            end_date (Optional[datetime]): 종료 날짜
            level (Optional[str]): 로그 레벨
            log_type (Optional[str]): 로그 유형
            
        Returns:
            List[Dict[str, Any]]: 시스템 로그 목록
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM system_logs WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                    
                if level:
                    query += " AND level = ?"
                    params.append(level)
                    
                if log_type:
                    query += " AND type = ?"
                    params.append(log_type)
                    
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"시스템 로그 조회 실패: {str(e)}")
            return []
            
    def save_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        알림 저장
        
        Args:
            notification_data (Dict[str, Any]): 알림 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO notifications (
                        timestamp, type, message, status, channel
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    notification_data['timestamp'],
                    notification_data['type'],
                    notification_data['message'],
                    notification_data['status'],
                    notification_data['channel']
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"알림 저장 실패: {str(e)}")
            return False
            
    def get_notifications(self,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         notification_type: Optional[str] = None,
                         status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        알림 조회
        
        Args:
            start_date (Optional[datetime]): 시작 날짜
            end_date (Optional[datetime]): 종료 날짜
            notification_type (Optional[str]): 알림 유형
            status (Optional[str]): 상태
            
        Returns:
            List[Dict[str, Any]]: 알림 목록
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM notifications WHERE 1=1"
                params = []
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                    
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                    
                if notification_type:
                    query += " AND type = ?"
                    params.append(notification_type)
                    
                if status:
                    query += " AND status = ?"
                    params.append(status)
                    
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"알림 조회 실패: {str(e)}")
            return []
            
    def export_data(self, table: str, format: str = "csv") -> Optional[str]:
        """
        데이터 내보내기
        
        Args:
            table (str): 테이블명
            format (str): 내보내기 형식 (csv/json)
            
        Returns:
            Optional[str]: 내보낸 데이터
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                
                if format == "csv":
                    return df.to_csv(index=False)
                else:
                    return df.to_json(orient="records")
                    
        except Exception as e:
            self.logger.error(f"데이터 내보내기 실패: {str(e)}")
            return None 