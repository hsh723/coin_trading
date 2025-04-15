"""
데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from src.analysis.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger
import json
import os
from datetime import datetime, timedelta
from scipy import stats
import talib
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

class DataProcessor:
    """데이터 전처리 클래스"""
    
    def __init__(self,
                 config_path: str = "./config/processor_config.json",
                 data_dir: str = "./data",
                 log_dir: str = "./logs"):
        """
        데이터 전처리기 초기화
        
        Args:
            config_path: 설정 파일 경로
            data_dir: 데이터 디렉토리
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 로거 설정
        self.logger = logging.getLogger("data_processor")
        
        # 설정 로드
        self.config = self._load_config()
        
        # 데이터 큐
        self.data_queue = queue.Queue()
        
        # 데이터베이스 연결
        self.db_path = os.path.join(data_dir, "processed_data.db")
        self._init_database()
        
        # 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.indicators = TechnicalIndicators()
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def _init_database(self) -> None:
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 처리된 데이터 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_data (
                    timestamp DATETIME,
                    symbol TEXT,
                    data_type TEXT,
                    feature_name TEXT,
                    value REAL,
                    PRIMARY KEY (timestamp, symbol, data_type, feature_name)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"데이터베이스 초기화 중 오류 발생: {e}")
            raise
            
    def process_data(self, data: Dict[str, Any]) -> None:
        """
        데이터 처리
        
        Args:
            data: 처리할 데이터
        """
        try:
            # 데이터 타입에 따른 처리
            if data["type"] == "trade":
                self._process_trade_data(data)
            elif data["type"] == "orderbook":
                self._process_orderbook_data(data)
            elif data["type"] == "ticker":
                self._process_ticker_data(data)
            elif data["type"] == "kline":
                self._process_kline_data(data)
                
        except Exception as e:
            self.logger.error(f"데이터 처리 중 오류 발생: {e}")
            
    def _process_trade_data(self, data: Dict[str, Any]) -> None:
        """
        거래 데이터 처리
        
        Args:
            data: 거래 데이터
        """
        try:
            # 데이터 추출
            timestamp = data["timestamp"]
            symbol = data["symbol"]
            price = data["price"]
            quantity = data["quantity"]
            side = data["side"]
            
            # 거래량 가중 평균 가격 계산
            vwap = self._calculate_vwap(timestamp, symbol, price, quantity)
            
            # 거래량 분포 계산
            volume_dist = self._calculate_volume_distribution(timestamp, symbol, quantity)
            
            # 데이터베이스 저장
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # VWAP 저장
            cursor.execute('''
                INSERT OR REPLACE INTO processed_data
                (timestamp, symbol, data_type, feature_name, value)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, symbol, "trade", "vwap", vwap))
            
            # 거래량 분포 저장
            for dist in volume_dist:
                cursor.execute('''
                    INSERT OR REPLACE INTO processed_data
                    (timestamp, symbol, data_type, feature_name, value)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, symbol, "trade", f"volume_dist_{dist['bucket']}", dist["value"]))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"거래 데이터 처리 중 오류 발생: {e}")
            
    def _process_orderbook_data(self, data: Dict[str, Any]) -> None:
        """
        호가 데이터 처리
        
        Args:
            data: 호가 데이터
        """
        try:
            # 데이터 추출
            timestamp = data["timestamp"]
            symbol = data["symbol"]
            bids = data["bids"]
            asks = data["asks"]
            
            # 호가 스프레드 계산
            spread = self._calculate_spread(bids, asks)
            
            # 호가 깊이 계산
            depth = self._calculate_depth(bids, asks)
            
            # 데이터베이스 저장
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 스프레드 저장
            cursor.execute('''
                INSERT OR REPLACE INTO processed_data
                (timestamp, symbol, data_type, feature_name, value)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, symbol, "orderbook", "spread", spread))
            
            # 깊이 저장
            for d in depth:
                cursor.execute('''
                    INSERT OR REPLACE INTO processed_data
                    (timestamp, symbol, data_type, feature_name, value)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, symbol, "orderbook", f"depth_{d['side']}", d["value"]))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"호가 데이터 처리 중 오류 발생: {e}")
            
    def _process_ticker_data(self, data: Dict[str, Any]) -> None:
        """
        티커 데이터 처리
        
        Args:
            data: 티커 데이터
        """
        try:
            # 데이터 추출
            timestamp = data["timestamp"]
            symbol = data["symbol"]
            last_price = data["last_price"]
            bid_price = data["bid_price"]
            ask_price = data["ask_price"]
            volume = data["volume"]
            
            # 가격 변동성 계산
            volatility = self._calculate_volatility(timestamp, symbol, last_price)
            
            # 거래량 추세 계산
            volume_trend = self._calculate_volume_trend(timestamp, symbol, volume)
            
            # 데이터베이스 저장
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 변동성 저장
            cursor.execute('''
                INSERT OR REPLACE INTO processed_data
                (timestamp, symbol, data_type, feature_name, value)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, symbol, "ticker", "volatility", volatility))
            
            # 거래량 추세 저장
            cursor.execute('''
                INSERT OR REPLACE INTO processed_data
                (timestamp, symbol, data_type, feature_name, value)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, symbol, "ticker", "volume_trend", volume_trend))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"티커 데이터 처리 중 오류 발생: {e}")
            
    def _process_kline_data(self, data: Dict[str, Any]) -> None:
        """
        캔들 데이터 처리
        
        Args:
            data: 캔들 데이터
        """
        try:
            # 데이터 추출
            timestamp = data["timestamp"]
            symbol = data["symbol"]
            interval = data["interval"]
            open_price = data["open"]
            high_price = data["high"]
            low_price = data["low"]
            close_price = data["close"]
            volume = data["volume"]
            
            # 기술적 지표 계산
            indicators = self._calculate_technical_indicators(
                open_price, high_price, low_price, close_price, volume
            )
            
            # 데이터베이스 저장
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 기술적 지표 저장
            for name, value in indicators.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO processed_data
                    (timestamp, symbol, data_type, feature_name, value)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, symbol, "kline", name, value))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"캔들 데이터 처리 중 오류 발생: {e}")
            
    def _calculate_vwap(self,
                       timestamp: datetime,
                       symbol: str,
                       price: float,
                       quantity: float) -> float:
        """
        VWAP 계산
        
        Args:
            timestamp: 타임스탬프
            symbol: 심볼
            price: 가격
            quantity: 수량
            
        Returns:
            VWAP
        """
        try:
            # 이전 VWAP 조회
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT value FROM processed_data
                WHERE symbol = ? AND data_type = 'trade'
                AND feature_name = 'vwap'
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                prev_vwap = result[0]
                total_volume = self._get_total_volume(timestamp, symbol)
                return (prev_vwap * total_volume + price * quantity) / (total_volume + quantity)
            else:
                return price
                
        except Exception as e:
            self.logger.error(f"VWAP 계산 중 오류 발생: {e}")
            return price
            
    def _calculate_volume_distribution(self,
                                     timestamp: datetime,
                                     symbol: str,
                                     quantity: float) -> List[Dict[str, Any]]:
        """
        거래량 분포 계산
        
        Args:
            timestamp: 타임스탬프
            symbol: 심볼
            quantity: 수량
            
        Returns:
            거래량 분포
        """
        try:
            # 거래량 버킷 정의
            buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
            
            # 분포 계산
            distribution = []
            total_volume = self._get_total_volume(timestamp, symbol)
            
            for i in range(len(buckets)):
                if i == 0:
                    min_volume = 0
                else:
                    min_volume = buckets[i-1]
                    
                max_volume = buckets[i]
                
                # 해당 버킷의 거래량 비율 계산
                bucket_volume = self._get_bucket_volume(timestamp, symbol, min_volume, max_volume)
                ratio = bucket_volume / total_volume if total_volume > 0 else 0
                
                distribution.append({
                    "bucket": f"{min_volume}-{max_volume}",
                    "value": ratio
                })
                
            return distribution
            
        except Exception as e:
            self.logger.error(f"거래량 분포 계산 중 오류 발생: {e}")
            return []
            
    def _calculate_spread(self,
                         bids: List[List[float]],
                         asks: List[List[float]]) -> float:
        """
        스프레드 계산
        
        Args:
            bids: 매수 호가
            asks: 매도 호가
            
        Returns:
            스프레드
        """
        try:
            if not bids or not asks:
                return 0
                
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            return (best_ask - best_bid) / best_bid * 100
            
        except Exception as e:
            self.logger.error(f"스프레드 계산 중 오류 발생: {e}")
            return 0
            
    def _calculate_depth(self,
                        bids: List[List[float]],
                        asks: List[List[float]]) -> List[Dict[str, Any]]:
        """
        호가 깊이 계산
        
        Args:
            bids: 매수 호가
            asks: 매도 호가
            
        Returns:
            호가 깊이
        """
        try:
            depth = []
            
            # 매수 호가 깊이
            bid_depth = sum(qty for _, qty in bids)
            depth.append({"side": "bid", "value": bid_depth})
            
            # 매도 호가 깊이
            ask_depth = sum(qty for _, qty in asks)
            depth.append({"side": "ask", "value": ask_depth})
            
            return depth
            
        except Exception as e:
            self.logger.error(f"호가 깊이 계산 중 오류 발생: {e}")
            return []
            
    def _calculate_volatility(self,
                            timestamp: datetime,
                            symbol: str,
                            price: float) -> float:
        """
        변동성 계산
        
        Args:
            timestamp: 타임스탬프
            symbol: 심볼
            price: 가격
            
        Returns:
            변동성
        """
        try:
            # 이전 가격 조회
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT value FROM processed_data
                WHERE symbol = ? AND data_type = 'ticker'
                AND feature_name = 'last_price'
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                prev_price = result[0]
                return abs((price - prev_price) / prev_price * 100)
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"변동성 계산 중 오류 발생: {e}")
            return 0
            
    def _calculate_volume_trend(self,
                              timestamp: datetime,
                              symbol: str,
                              volume: float) -> float:
        """
        거래량 추세 계산
        
        Args:
            timestamp: 타임스탬프
            symbol: 심볼
            volume: 거래량
            
        Returns:
            거래량 추세
        """
        try:
            # 이전 거래량 조회
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT value FROM processed_data
                WHERE symbol = ? AND data_type = 'ticker'
                AND feature_name = 'volume'
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                prev_volume = result[0]
                return (volume - prev_volume) / prev_volume * 100 if prev_volume > 0 else 0
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"거래량 추세 계산 중 오류 발생: {e}")
            return 0
            
    def _calculate_technical_indicators(self,
                                      open_price: float,
                                      high_price: float,
                                      low_price: float,
                                      close_price: float,
                                      volume: float) -> Dict[str, float]:
        """
        기술적 지표 계산
        
        Args:
            open_price: 시가
            high_price: 고가
            low_price: 저가
            close_price: 종가
            volume: 거래량
            
        Returns:
            기술적 지표
        """
        try:
            indicators = {}
            
            # RSI
            rsi = talib.RSI(np.array([close_price]), timeperiod=14)[-1]
            indicators["rsi"] = rsi
            
            # MACD
            macd, signal, hist = talib.MACD(
                np.array([close_price]),
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            indicators["macd"] = macd[-1]
            indicators["macd_signal"] = signal[-1]
            indicators["macd_hist"] = hist[-1]
            
            # 볼린저 밴드
            upper, middle, lower = talib.BBANDS(
                np.array([close_price]),
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            indicators["bb_upper"] = upper[-1]
            indicators["bb_middle"] = middle[-1]
            indicators["bb_lower"] = lower[-1]
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 중 오류 발생: {e}")
            return {}
            
    def _get_total_volume(self,
                         timestamp: datetime,
                         symbol: str) -> float:
        """
        총 거래량 조회
        
        Args:
            timestamp: 타임스탬프
            symbol: 심볼
            
        Returns:
            총 거래량
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT SUM(value) FROM processed_data
                WHERE symbol = ? AND data_type = 'trade'
                AND feature_name = 'quantity'
                AND timestamp >= ?
            ''', (symbol, timestamp - timedelta(hours=1)))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] else 0
            
        except Exception as e:
            self.logger.error(f"총 거래량 조회 중 오류 발생: {e}")
            return 0
            
    def _get_bucket_volume(self,
                          timestamp: datetime,
                          symbol: str,
                          min_volume: float,
                          max_volume: float) -> float:
        """
        버킷 거래량 조회
        
        Args:
            timestamp: 타임스탬프
            symbol: 심볼
            min_volume: 최소 거래량
            max_volume: 최대 거래량
            
        Returns:
            버킷 거래량
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT SUM(value) FROM processed_data
                WHERE symbol = ? AND data_type = 'trade'
                AND feature_name = 'quantity'
                AND value >= ? AND value < ?
                AND timestamp >= ?
            ''', (symbol, min_volume, max_volume, timestamp - timedelta(hours=1)))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] else 0
            
        except Exception as e:
            self.logger.error(f"버킷 거래량 조회 중 오류 발생: {e}")
            return 0
            
    def get_processed_data(self,
                          symbol: str,
                          data_type: str,
                          feature_name: str,
                          start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """
        처리된 데이터 조회
        
        Args:
            symbol: 심볼
            data_type: 데이터 타입
            feature_name: 피처 이름
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            데이터프레임
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, value FROM processed_data
                WHERE symbol = ? AND data_type = ? AND feature_name = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, data_type, feature_name, start_time, end_time)
            )
            
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"처리된 데이터 조회 중 오류 발생: {e}")
            raise

# 전역 인스턴스 생성
data_processor = DataProcessor() 