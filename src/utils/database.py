"""
데이터베이스 관리 모듈
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
from .logger import logger

class DatabaseManager:
    def __init__(self, db_path: str = 'data/trading.db'):
        """데이터베이스 관리자 초기화"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 거래 테이블 생성
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        type TEXT NOT NULL,
                        side TEXT NOT NULL,
                        price REAL NOT NULL,
                        amount REAL NOT NULL,
                        cost REAL NOT NULL,
                        fee REAL NOT NULL,
                        pnl REAL
                    )
                ''')
                
                # 성과 테이블 생성
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        daily_return REAL,
                        weekly_return REAL,
                        monthly_return REAL,
                        total_trades INTEGER,
                        total_pnl REAL
                    )
                ''')
                
                # 사용자 테이블 생성
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        role TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        last_login DATETIME,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # 사용자 세션 테이블 생성
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        token TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        expires_at DATETIME NOT NULL,
                        last_activity DATETIME NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                conn.commit()
                logger.info("데이터베이스 초기화 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            raise
    
    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """거래 정보 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO trades (
                        timestamp, symbol, type, side,
                        price, amount, cost, fee, pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['timestamp'],
                    trade_data['symbol'],
                    trade_data['type'],
                    trade_data['side'],
                    trade_data['price'],
                    trade_data['amount'],
                    trade_data['cost'],
                    trade_data['fee'],
                    trade_data.get('pnl')
                ))
                conn.commit()
                logger.info(f"거래 정보 저장 완료: {trade_data['symbol']}")
                return True
        except Exception as e:
            logger.error(f"거래 정보 저장 실패: {str(e)}")
            return False
    
    def save_performance(self, performance_data: Dict[str, Any]) -> bool:
        """성과 정보 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO performance (
                        date, daily_return, weekly_return,
                        monthly_return, total_trades, total_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    performance_data['date'],
                    performance_data['daily_return'],
                    performance_data['weekly_return'],
                    performance_data['monthly_return'],
                    performance_data['total_trades'],
                    performance_data['total_pnl']
                ))
                conn.commit()
                logger.info(f"성과 정보 저장 완료: {performance_data['date']}")
                return True
        except Exception as e:
            logger.error(f"성과 정보 저장 실패: {str(e)}")
            return False
    
    def get_trades(self, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """거래 정보 조회"""
        try:
            query = "SELECT * FROM trades"
            params = []
            
            if start_date and end_date:
                query += " WHERE timestamp BETWEEN ? AND ?"
                params.extend([start_date, end_date])
            elif start_date:
                query += " WHERE timestamp >= ?"
                params.append(start_date)
            elif end_date:
                query += " WHERE timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
        except Exception as e:
            logger.error(f"거래 정보 조회 실패: {str(e)}")
            return pd.DataFrame()
    
    def get_performance(self, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """성과 정보 조회"""
        try:
            query = "SELECT * FROM performance"
            params = []
            
            if start_date and end_date:
                query += " WHERE date BETWEEN ? AND ?"
                params.extend([start_date, end_date])
            elif start_date:
                query += " WHERE date >= ?"
                params.append(start_date)
            elif end_date:
                query += " WHERE date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
        except Exception as e:
            logger.error(f"성과 정보 조회 실패: {str(e)}")
            return pd.DataFrame()
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """사용자 정보 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'username': row[1],
                        'password': row[2],
                        'role': row[3],
                        'created_at': row[4],
                        'last_login': row[5],
                        'is_active': row[6]
                    }
                return None
        except Exception as e:
            logger.error(f"사용자 정보 조회 실패: {str(e)}")
            return None
    
    def save_user(self, user_data: Dict[str, Any]) -> bool:
        """사용자 정보 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO users (
                        username, password, role,
                        created_at, last_login
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    user_data['username'],
                    user_data['password'],
                    user_data['role'],
                    user_data['created_at'],
                    user_data.get('last_login')
                ))
                conn.commit()
                logger.info(f"사용자 정보 저장 완료: {user_data['username']}")
                return True
        except Exception as e:
            logger.error(f"사용자 정보 저장 실패: {str(e)}")
            return False
    
    def update_user_last_login(self, username: str) -> bool:
        """사용자 마지막 로그인 시간 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE users
                    SET last_login = ?
                    WHERE username = ?
                ''', (datetime.now(), username))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"사용자 로그인 시간 업데이트 실패: {str(e)}")
            return False
    
    def save_session(self, user_id: int, token: str,
                    expires_at: datetime) -> bool:
        """사용자 세션 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO user_sessions (
                        user_id, token, created_at,
                        expires_at, last_activity
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    token,
                    datetime.now(),
                    expires_at,
                    datetime.now()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"세션 저장 실패: {str(e)}")
            return False
    
    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM user_sessions WHERE token = ?",
                    (token,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'user_id': row[1],
                        'token': row[2],
                        'created_at': row[3],
                        'expires_at': row[4],
                        'last_activity': row[5]
                    }
                return None
        except Exception as e:
            logger.error(f"세션 정보 조회 실패: {str(e)}")
            return None
    
    def update_session_activity(self, token: str) -> bool:
        """세션 활동 시간 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE user_sessions
                    SET last_activity = ?
                    WHERE token = ?
                ''', (datetime.now(), token))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"세션 활동 시간 업데이트 실패: {str(e)}")
            return False
    
    def delete_expired_sessions(self) -> bool:
        """만료된 세션 삭제"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    DELETE FROM user_sessions
                    WHERE expires_at < ?
                ''', (datetime.now(),))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"만료된 세션 삭제 실패: {str(e)}")
            return False

# 전역 데이터베이스 관리자 인스턴스
db_manager = DatabaseManager() 