import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from ..utils.logger import setup_logger

class Database:
    """데이터베이스 클래스"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        """
        데이터베이스 초기화
        
        Args:
            db_path (str): 데이터베이스 파일 경로
        """
        self.logger = setup_logger('database')
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self) -> None:
        """데이터베이스 테이블 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 거래 내역 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        price REAL NOT NULL,
                        amount REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        order_id TEXT,
                        pnl REAL,
                        status TEXT NOT NULL
                    )
                ''')
                
                # 포지션 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        amount REAL NOT NULL,
                        stop_loss REAL,
                        take_profit REAL,
                        entry_time DATETIME NOT NULL,
                        exit_time DATETIME,
                        exit_price REAL,
                        pnl REAL,
                        status TEXT NOT NULL
                    )
                ''')
                
                # 성과 지표 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        daily_return REAL,
                        weekly_return REAL,
                        monthly_return REAL,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        win_rate REAL,
                        total_pnl REAL,
                        max_drawdown REAL
                    )
                ''')
                
                # 설정 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL UNIQUE,
                        value TEXT NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                ''')
                
                # 로그 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        source TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                self.logger.info("데이터베이스 초기화 완료")
                
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            
    def save_trade(self, trade: Dict[str, Any]) -> None:
        """
        거래 내역 저장
        
        Args:
            trade (Dict[str, Any]): 거래 정보
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (
                        symbol, side, price, amount, timestamp,
                        order_id, pnl, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade['symbol'],
                    trade['side'],
                    trade['price'],
                    trade['amount'],
                    trade.get('timestamp', datetime.now()),
                    trade.get('order_id'),
                    trade.get('pnl'),
                    trade.get('status', 'open')
                ))
                conn.commit()
                self.logger.info(f"거래 내역 저장 완료: {trade['symbol']}")
                
        except Exception as e:
            self.logger.error(f"거래 내역 저장 실패: {str(e)}")
            
    def save_position(self, position: Dict[str, Any]) -> None:
        """
        포지션 정보 저장
        
        Args:
            position (Dict[str, Any]): 포지션 정보
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO positions (
                        symbol, entry_price, amount, stop_loss,
                        take_profit, entry_time, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position['symbol'],
                    position['entry_price'],
                    position['amount'],
                    position.get('stop_loss'),
                    position.get('take_profit'),
                    position.get('entry_time', datetime.now()),
                    position.get('status', 'open')
                ))
                conn.commit()
                self.logger.info(f"포지션 정보 저장 완료: {position['symbol']}")
                
        except Exception as e:
            self.logger.error(f"포지션 정보 저장 실패: {str(e)}")
            
    def update_position(self, position_id: int, updates: Dict[str, Any]) -> None:
        """
        포지션 정보 업데이트
        
        Args:
            position_id (int): 포지션 ID
            updates (Dict[str, Any]): 업데이트할 정보
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values())
                values.append(position_id)
                
                cursor.execute(f'''
                    UPDATE positions
                    SET {set_clause}
                    WHERE id = ?
                ''', values)
                
                conn.commit()
                self.logger.info(f"포지션 정보 업데이트 완료: ID {position_id}")
                
        except Exception as e:
            self.logger.error(f"포지션 정보 업데이트 실패: {str(e)}")
            
    def save_performance(self, performance: Dict[str, Any]) -> None:
        """
        성과 지표 저장
        
        Args:
            performance (Dict[str, Any]): 성과 지표
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance (
                        date, daily_return, weekly_return, monthly_return,
                        total_trades, winning_trades, losing_trades,
                        win_rate, total_pnl, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.get('date', datetime.now().date()),
                    performance.get('daily_return'),
                    performance.get('weekly_return'),
                    performance.get('monthly_return'),
                    performance.get('total_trades', 0),
                    performance.get('winning_trades', 0),
                    performance.get('losing_trades', 0),
                    performance.get('win_rate', 0.0),
                    performance.get('total_pnl', 0.0),
                    performance.get('max_drawdown', 0.0)
                ))
                conn.commit()
                self.logger.info("성과 지표 저장 완료")
                
        except Exception as e:
            self.logger.error(f"성과 지표 저장 실패: {str(e)}")
            
    def save_setting(self, key: str, value: str) -> None:
        """
        설정 저장
        
        Args:
            key (str): 설정 키
            value (str): 설정 값
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                ''', (key, value, datetime.now()))
                conn.commit()
                self.logger.info(f"설정 저장 완료: {key}")
                
        except Exception as e:
            self.logger.error(f"설정 저장 실패: {str(e)}")
            
    def get_setting(self, key: str) -> Optional[str]:
        """
        설정 조회
        
        Args:
            key (str): 설정 키
            
        Returns:
            Optional[str]: 설정 값
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT value FROM settings WHERE key = ?
                ''', (key,))
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            self.logger.error(f"설정 조회 실패: {str(e)}")
            return None
            
    def save_log(self, level: str, message: str, source: str) -> None:
        """
        로그 저장
        
        Args:
            level (str): 로그 레벨
            message (str): 로그 메시지
            source (str): 로그 소스
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO logs (timestamp, level, message, source)
                    VALUES (?, ?, ?, ?)
                ''', (datetime.now(), level, message, source))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"로그 저장 실패: {str(e)}")
            
    def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        거래 내역 조회
        
        Args:
            limit (int): 조회할 최대 개수
            
        Returns:
            List[Dict[str, Any]]: 거래 내역 리스트
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM trades
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"거래 내역 조회 실패: {str(e)}")
            return []
            
    def get_positions(self, status: str = None) -> List[Dict[str, Any]]:
        """
        포지션 정보 조회
        
        Args:
            status (str): 포지션 상태 필터
            
        Returns:
            List[Dict[str, Any]]: 포지션 정보 리스트
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if status:
                    cursor.execute('''
                        SELECT * FROM positions
                        WHERE status = ?
                        ORDER BY entry_time DESC
                    ''', (status,))
                else:
                    cursor.execute('''
                        SELECT * FROM positions
                        ORDER BY entry_time DESC
                    ''')
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"포지션 정보 조회 실패: {str(e)}")
            return []
            
    def get_performance(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        성과 지표 조회
        
        Args:
            start_date (datetime): 시작 날짜
            end_date (datetime): 종료 날짜
            
        Returns:
            List[Dict[str, Any]]: 성과 지표 리스트
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM performance
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date DESC
                ''', (start_date.date(), end_date.date()))
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"성과 지표 조회 실패: {str(e)}")
            return []
            
    def get_logs(self, level: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        로그 조회
        
        Args:
            level (str): 로그 레벨 필터
            limit (int): 조회할 최대 개수
            
        Returns:
            List[Dict[str, Any]]: 로그 리스트
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if level:
                    cursor.execute('''
                        SELECT * FROM logs
                        WHERE level = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (level, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM logs
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"로그 조회 실패: {str(e)}")
            return [] 