"""
데이터 저장 및 로드 모듈
"""

import os
import pandas as pd
import json
import gzip
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from ..utils.logger import setup_logger
from ..utils.config_loader import get_config

class DataStorage:
    """
    데이터 저장 및 로드를 관리하는 클래스
    """
    
    def __init__(self, base_dir: str = 'data'):
        """
        초기화
        
        Args:
            base_dir (str): 데이터 저장 기본 디렉토리
        """
        self.logger = setup_logger()
        self.base_dir = Path(base_dir)
        
        # 디렉토리 구조 생성
        self.dirs = {
            'ohlcv': self.base_dir / 'ohlcv',
            'backtest': self.base_dir / 'backtest',
            'logs': self.base_dir / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DataStorage initialized with base directory: {base_dir}")
    
    def save_ohlcv(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        compressed: bool = True
    ) -> str:
        """
        OHLCV 데이터 저장
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            compressed (bool): 압축 여부
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{timeframe}_{timestamp}"
            
            # 저장 경로 설정
            if compressed:
                filepath = self.dirs['ohlcv'] / f"{filename}.csv.gz"
                # CSV로 저장 후 압축
                data.to_csv(filepath, compression='gzip')
            else:
                filepath = self.dirs['ohlcv'] / f"{filename}.csv"
                data.to_csv(filepath)
            
            self.logger.info(f"Saved OHLCV data to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving OHLCV data: {str(e)}")
            raise
    
    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        compressed: bool = True
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 로드
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            start_date (datetime, optional): 시작 날짜
            end_date (datetime, optional): 종료 날짜
            compressed (bool): 압축 여부
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            # 파일 패턴 생성
            pattern = f"{symbol}_{timeframe}_*.csv"
            if compressed:
                pattern += ".gz"
            
            # 파일 검색
            files = list(self.dirs['ohlcv'].glob(pattern))
            if not files:
                raise FileNotFoundError(f"No OHLCV data found for {symbol} ({timeframe})")
            
            # 가장 최근 파일 선택
            latest_file = max(files, key=os.path.getctime)
            
            # 데이터 로드
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            
            # 날짜 필터링
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            self.logger.info(f"Loaded OHLCV data from {latest_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading OHLCV data: {str(e)}")
            raise
    
    def save_backtest_result(
        self,
        result: Dict,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        compressed: bool = True
    ) -> str:
        """
        백테스트 결과 저장
        
        Args:
            result (Dict): 백테스트 결과
            strategy_name (str): 전략 이름
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            compressed (bool): 압축 여부
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{strategy_name}_{symbol}_{timeframe}_{timestamp}"
            
            # 저장 경로 설정
            if compressed:
                filepath = self.dirs['backtest'] / f"{filename}.json.gz"
                # JSON으로 변환 후 압축
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                filepath = self.dirs['backtest'] / f"{filename}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved backtest result to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving backtest result: {str(e)}")
            raise
    
    def load_backtest_result(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        compressed: bool = True
    ) -> Dict:
        """
        백테스트 결과 로드
        
        Args:
            strategy_name (str): 전략 이름
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            compressed (bool): 압축 여부
            
        Returns:
            Dict: 백테스트 결과
        """
        try:
            # 파일 패턴 생성
            pattern = f"{strategy_name}_{symbol}_{timeframe}_*.json"
            if compressed:
                pattern += ".gz"
            
            # 파일 검색
            files = list(self.dirs['backtest'].glob(pattern))
            if not files:
                raise FileNotFoundError(f"No backtest result found for {strategy_name} ({symbol}, {timeframe})")
            
            # 가장 최근 파일 선택
            latest_file = max(files, key=os.path.getctime)
            
            # 데이터 로드
            if compressed:
                with gzip.open(latest_file, 'rt', encoding='utf-8') as f:
                    result = json.load(f)
            else:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            
            self.logger.info(f"Loaded backtest result from {latest_file}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading backtest result: {str(e)}")
            raise
    
    def save_trade_log(
        self,
        trades: List[Dict],
        symbol: str,
        compressed: bool = True
    ) -> str:
        """
        거래 로그 저장
        
        Args:
            trades (List[Dict]): 거래 기록 리스트
            symbol (str): 거래 심볼
            compressed (bool): 압축 여부
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trades_{symbol}_{timestamp}"
            
            # DataFrame 생성
            df = pd.DataFrame(trades)
            
            # 저장 경로 설정
            if compressed:
                filepath = self.dirs['logs'] / f"{filename}.csv.gz"
                # CSV로 저장 후 압축
                df.to_csv(filepath, compression='gzip')
            else:
                filepath = self.dirs['logs'] / f"{filename}.csv"
                df.to_csv(filepath)
            
            self.logger.info(f"Saved trade log to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving trade log: {str(e)}")
            raise
    
    def load_trade_log(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        compressed: bool = True
    ) -> pd.DataFrame:
        """
        거래 로그 로드
        
        Args:
            symbol (str): 거래 심볼
            start_date (datetime, optional): 시작 날짜
            end_date (datetime, optional): 종료 날짜
            compressed (bool): 압축 여부
            
        Returns:
            pd.DataFrame: 거래 로그
        """
        try:
            # 파일 패턴 생성
            pattern = f"trades_{symbol}_*.csv"
            if compressed:
                pattern += ".gz"
            
            # 파일 검색
            files = list(self.dirs['logs'].glob(pattern))
            if not files:
                raise FileNotFoundError(f"No trade log found for {symbol}")
            
            # 가장 최근 파일 선택
            latest_file = max(files, key=os.path.getctime)
            
            # 데이터 로드
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            
            # 날짜 필터링
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            self.logger.info(f"Loaded trade log from {latest_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading trade log: {str(e)}")
            raise 