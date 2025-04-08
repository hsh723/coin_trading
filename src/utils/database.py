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

# 로거 설정
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = 'data/trading.db'):
        """데이터베이스 관리자 초기화"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # 거래 내역 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0.0,
                    pnl REAL DEFAULT 0.0
                )
            """)
            
            # 포지션 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0.0,
                    last_update DATETIME NOT NULL
                )
            """)
            
            # 계좌 잔고 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS balance (
                    timestamp DATETIME PRIMARY KEY,
                    total_balance REAL NOT NULL,
                    available_balance REAL NOT NULL,
                    locked_balance REAL NOT NULL
                )
            """)
            
            # 성과 지표 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    timestamp DATETIME PRIMARY KEY,
                    total_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL
                )
            """)
            
            self.conn.commit()
            logger.info("데이터베이스 초기화 완료")
            
        except sqlite3.Error as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    def close(self):
        """연결 종료"""
        if self.conn:
            self.conn.close()
    
    def insert_trade(self, trade_data: Dict):
        """거래 내역 추가"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, symbol, side, quantity, 
                    price, commission, pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['timestamp'],
                trade_data['symbol'],
                trade_data['side'],
                trade_data['quantity'],
                trade_data['price'],
                trade_data.get('commission', 0.0),
                trade_data.get('pnl', 0.0)
            ))
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"거래 내역 추가 실패: {e}")
            raise
    
    def update_position(self, position_data: Dict):
        """포지션 업데이트"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO positions (
                    symbol, quantity, entry_price, 
                    current_price, unrealized_pnl, last_update
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                position_data['symbol'],
                position_data['quantity'],
                position_data['entry_price'],
                position_data['current_price'],
                position_data.get('unrealized_pnl', 0.0),
                position_data['last_update']
            ))
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"포지션 업데이트 실패: {e}")
            raise
    
    def update_balance(self, balance_data: Dict):
        """잔고 업데이트"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO balance (
                    timestamp, total_balance, 
                    available_balance, locked_balance
                ) VALUES (?, ?, ?, ?)
            """, (
                balance_data['timestamp'],
                balance_data['total_balance'],
                balance_data['available_balance'],
                balance_data['locked_balance']
            ))
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"잔고 업데이트 실패: {e}")
            raise
    
    def update_performance(self, performance_data: Dict):
        """성과 지표 업데이트"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance (
                    timestamp, total_pnl, win_rate,
                    profit_factor, sharpe_ratio, max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                performance_data['timestamp'],
                performance_data['total_pnl'],
                performance_data['win_rate'],
                performance_data['profit_factor'],
                performance_data['sharpe_ratio'],
                performance_data['max_drawdown']
            ))
            self.conn.commit()
            
        except sqlite3.Error as e:
            logger.error(f"성과 지표 업데이트 실패: {e}")
            raise
    
    def get_trades(self, 
                  symbol: Optional[str] = None, 
                  start_time: Optional[str] = None,
                  end_time: Optional[str] = None) -> pd.DataFrame:
        """거래 내역 조회"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        try:
            return pd.read_sql_query(query, self.conn, params=params)
        except pd.io.sql.DatabaseError as e:
            logger.error(f"거래 내역 조회 실패: {e}")
            return pd.DataFrame()
            
    def get_positions(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """포지션 조회"""
        query = "SELECT * FROM positions"
        params = None
        
        if symbol:
            query += " WHERE symbol = ?"
            params = [symbol]
            
        try:
            return pd.read_sql_query(query, self.conn, params=params)
        except pd.io.sql.DatabaseError as e:
            logger.error(f"포지션 조회 실패: {e}")
            return pd.DataFrame()
            
    def get_balance_history(self, 
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> pd.DataFrame:
        """잔고 내역 조회"""
        query = "SELECT * FROM balance WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        try:
            return pd.read_sql_query(query, self.conn, params=params)
        except pd.io.sql.DatabaseError as e:
            logger.error(f"잔고 내역 조회 실패: {e}")
            return pd.DataFrame()
            
    def get_performance_history(self,
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None) -> pd.DataFrame:
        """성과 지표 내역 조회"""
        query = "SELECT * FROM performance WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        try:
            return pd.read_sql_query(query, self.conn, params=params)
        except pd.io.sql.DatabaseError as e:
            logger.error(f"성과 지표 내역 조회 실패: {e}")
            return pd.DataFrame()

# 전역 데이터베이스 관리자 인스턴스
db_manager = DatabaseManager() 