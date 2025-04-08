"""
데이터베이스 관리 모듈
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import json
import os
from pathlib import Path

from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: Optional[str] = None):
        """데이터베이스 관리자 초기화"""
        if db_path is None:
            # 프로젝트 루트 디렉토리에 data 폴더 생성
            self.db_path = str(Path(__file__).parent.parent.parent / 'data' / 'trading.db')
        else:
            self.db_path = db_path
            
        # 데이터베이스 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 데이터베이스 초기화
        self._init_db()
        
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self.backup_dir = self.config['database']['backup_dir']
        self.connection = None
        self.cursor = None
        
        # 데이터베이스 디렉토리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def _init_db(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 가격 데이터 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS price_data (
                        symbol TEXT,
                        timeframe TEXT,
                        timestamp INTEGER,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        PRIMARY KEY (symbol, timeframe, timestamp)
                    )
                """)
                
                # 기술적 지표 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS technical_indicators (
                        symbol TEXT,
                        timeframe TEXT,
                        timestamp INTEGER,
                        indicator_name TEXT,
                        indicator_value TEXT,
                        PRIMARY KEY (symbol, timeframe, timestamp, indicator_name)
                    )
                """)
                
                # A/B 테스트 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ab_tests (
                        test_name TEXT PRIMARY KEY,
                        control_strategy TEXT,
                        treatment_strategy TEXT,
                        start_time INTEGER,
                        end_time INTEGER,
                        sample_size INTEGER,
                        status TEXT
                    )
                """)
                
                # A/B 테스트 데이터 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ab_test_data (
                        test_name TEXT,
                        timestamp INTEGER,
                        group_name TEXT,
                        pnl REAL,
                        cumulative_pnl REAL,
                        PRIMARY KEY (test_name, timestamp, group_name)
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            raise
            
    def get_price_data(self,
                      symbol: str,
                      timeframe: str,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        가격 데이터 조회
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 가격 데이터
        """
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(int(start_time.timestamp() * 1000))
            if end_time:
                query += " AND timestamp <= ?"
                params.append(int(end_time.timestamp() * 1000))
                
            query += " ORDER BY timestamp ASC"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
            return df
            
        except Exception as e:
            self.logger.error(f"가격 데이터 조회 실패: {str(e)}")
            return pd.DataFrame()
            
    def save_price_data(self,
                       symbol: str,
                       timeframe: str,
                       data: pd.DataFrame):
        """
        가격 데이터 저장
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            data (pd.DataFrame): 가격 데이터
        """
        try:
            data = data.copy()
            data['symbol'] = symbol
            data['timeframe'] = timeframe
            data['timestamp'] = data.index.astype(np.int64) // 10**6
            
            with sqlite3.connect(self.db_path) as conn:
                data.to_sql('price_data', conn, if_exists='append', index=False)
                
        except Exception as e:
            self.logger.error(f"가격 데이터 저장 실패: {str(e)}")
            
    def get_technical_indicators(self,
                               symbol: str,
                               timeframe: str,
                               indicator_name: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> Dict:
        """
        기술적 지표 조회
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            indicator_name (Optional[str]): 지표 이름
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            Dict: 기술적 지표
        """
        try:
            query = """
                SELECT timestamp, indicator_name, indicator_value
                FROM technical_indicators
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if indicator_name:
                query += " AND indicator_name = ?"
                params.append(indicator_name)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(int(start_time.timestamp() * 1000))
            if end_time:
                query += " AND timestamp <= ?"
                params.append(int(end_time.timestamp() * 1000))
                
            query += " ORDER BY timestamp ASC"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            if df.empty:
                return {}
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            results = {}
            for name, group in df.groupby('indicator_name'):
                results[name] = group['indicator_value'].apply(json.loads)
                
            return results
            
        except Exception as e:
            self.logger.error(f"기술적 지표 조회 실패: {str(e)}")
            return {}
            
    def save_technical_indicators(self,
                                symbol: str,
                                timeframe: str,
                                indicators: Dict):
        """
        기술적 지표 저장
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            indicators (Dict): 기술적 지표
        """
        try:
            data = []
            timestamp = int(datetime.now().timestamp() * 1000)
            
            for name, values in indicators.items():
                if isinstance(values, (pd.Series, pd.DataFrame)):
                    values = values.to_dict()
                data.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': timestamp,
                    'indicator_name': name,
                    'indicator_value': json.dumps(values)
                })
                
            if data:
                df = pd.DataFrame(data)
                with sqlite3.connect(self.db_path) as conn:
                    df.to_sql('technical_indicators', conn, if_exists='append', index=False)
                    
        except Exception as e:
            self.logger.error(f"기술적 지표 저장 실패: {str(e)}")
            
    def save_ab_test_config(self, test_name: str, config: Dict):
        """
        A/B 테스트 설정 저장
        
        Args:
            test_name (str): 테스트 이름
            config (Dict): 테스트 설정
        """
        try:
            config = config.copy()
            config['test_name'] = test_name
            config['start_time'] = int(config['start_time'].timestamp() * 1000)
            config['end_time'] = int(config['end_time'].timestamp() * 1000)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.DataFrame([config])
                df.to_sql('ab_tests', conn, if_exists='replace', index=False)
                
        except Exception as e:
            self.logger.error(f"A/B 테스트 설정 저장 실패: {str(e)}")
            
    def update_ab_test_status(self, test_name: str, status: str):
        """
        A/B 테스트 상태 업데이트
        
        Args:
            test_name (str): 테스트 이름
            status (str): 테스트 상태
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE ab_tests SET status = ? WHERE test_name = ?",
                    (status, test_name)
                )
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"A/B 테스트 상태 업데이트 실패: {str(e)}")
            
    def get_ab_test_data(self, test_name: str) -> List[Dict]:
        """
        A/B 테스트 데이터 조회
        
        Args:
            test_name (str): 테스트 이름
            
        Returns:
            List[Dict]: 테스트 데이터
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM ab_test_data WHERE test_name = ?",
                    conn,
                    params=[test_name]
                )
                
            if df.empty:
                return []
                
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"A/B 테스트 데이터 조회 실패: {str(e)}")
            return []

# 전역 인스턴스 생성
database_manager = DatabaseManager() 