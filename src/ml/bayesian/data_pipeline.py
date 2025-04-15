import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime, timedelta
import logging
from kafka import KafkaProducer, KafkaConsumer
import websockets
import time
import pytz

class DataPipeline:
    """데이터 파이프라인"""
    def __init__(self,
                 config_path: str = "./config/data_pipeline_config.json",
                 data_dir: str = "./data",
                 log_dir: str = "./logs"):
        """
        데이터 파이프라인 초기화
        
        Args:
            config_path: 설정 파일 경로
            data_dir: 데이터 디렉토리
            log_dir: 로그 디렉토리
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # 설정 로드
        self.config = self._load_config()
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        # 데이터 버퍼
        self.data_buffer = {}
        
        # Kafka 프로듀서/컨슈머
        self.producer = None
        self.consumer = None
        
        # 웹소켓 연결
        self.ws = None
        
        # 데이터 처리 태스크
        self.processing_task = None
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("data_pipeline")
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        log_file = os.path.join(self.log_dir, "data_pipeline.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
        
    async def start(self):
        """데이터 파이프라인 시작"""
        # Kafka 연결
        self._setup_kafka()
        
        # 웹소켓 연결
        await self._connect_websocket()
        
        # 데이터 처리 시작
        self.processing_task = asyncio.create_task(self._process_data())
        self.logger.info("데이터 파이프라인 시작")
        
    async def stop(self):
        """데이터 파이프라인 중지"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
                
        if self.ws:
            await self.ws.close()
            
        if self.producer:
            self.producer.close()
            
        if self.consumer:
            self.consumer.close()
            
        self.logger.info("데이터 파이프라인 중지")
        
    def _setup_kafka(self):
        """Kafka 설정"""
        try:
            # 프로듀서 설정
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.get('kafka', {}).get('bootstrap_servers', ['localhost:9092']),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            # 컨슈머 설정
            self.consumer = KafkaConsumer(
                self.config.get('kafka', {}).get('topic', 'market_data'),
                bootstrap_servers=self.config.get('kafka', {}).get('bootstrap_servers', ['localhost:9092']),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
        except Exception as e:
            self.logger.error(f"Kafka 설정 중 오류 발생: {e}")
            raise
            
    async def _connect_websocket(self):
        """웹소켓 연결"""
        try:
            self.ws = await websockets.connect(
                self.config.get('websocket', {}).get('url', 'ws://localhost:8080')
            )
            self.logger.info("웹소켓 연결 성공")
        except Exception as e:
            self.logger.error(f"웹소켓 연결 중 오류 발생: {e}")
            raise
            
    async def _process_data(self):
        """데이터 처리"""
        while True:
            try:
                # 웹소켓에서 데이터 수신
                data = await self.ws.recv()
                data = json.loads(data)
                
                # 데이터 전처리
                processed_data = self._preprocess_data(data)
                
                # 데이터 버퍼에 저장
                self._update_buffer(processed_data)
                
                # Kafka에 전송
                await self._send_to_kafka(processed_data)
                
                # 데이터 저장
                self._save_data(processed_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"데이터 처리 중 오류 발생: {e}")
                
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 전처리"""
        try:
            # 타임스탬프 변환
            data['timestamp'] = datetime.fromtimestamp(
                data['timestamp'] / 1000,
                tz=pytz.UTC
            ).isoformat()
            
            # 숫자형 데이터 변환
            numeric_fields = ['price', 'volume', 'bid', 'ask']
            for field in numeric_fields:
                if field in data:
                    data[field] = float(data[field])
                    
            # 추가 필드 계산
            if 'price' in data and 'volume' in data:
                data['value'] = data['price'] * data['volume']
                
            return data
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류 발생: {e}")
            raise
            
    def _update_buffer(self, data: Dict[str, Any]):
        """데이터 버퍼 업데이트"""
        symbol = data.get('symbol')
        if not symbol:
            return
            
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
            
        self.data_buffer[symbol].append(data)
        
        # 버퍼 크기 제한
        max_buffer_size = self.config.get('buffer', {}).get('max_size', 1000)
        if len(self.data_buffer[symbol]) > max_buffer_size:
            self.data_buffer[symbol] = self.data_buffer[symbol][-max_buffer_size:]
            
    async def _send_to_kafka(self, data: Dict[str, Any]):
        """Kafka에 데이터 전송"""
        try:
            topic = self.config.get('kafka', {}).get('topic', 'market_data')
            self.producer.send(topic, value=data)
            self.producer.flush()
            
        except Exception as e:
            self.logger.error(f"Kafka 전송 중 오류 발생: {e}")
            raise
            
    def _save_data(self, data: Dict[str, Any]):
        """데이터 저장"""
        try:
            symbol = data.get('symbol')
            if not symbol:
                return
                
            # 데이터 디렉토리 생성
            symbol_dir = os.path.join(self.data_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # 일별 파일 경로
            date_str = datetime.fromisoformat(data['timestamp']).strftime('%Y%m%d')
            file_path = os.path.join(symbol_dir, f"{date_str}.csv")
            
            # 데이터프레임 생성
            df = pd.DataFrame([data])
            
            # 파일 존재 여부에 따라 저장 방식 결정
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, index=False)
                
        except Exception as e:
            self.logger.error(f"데이터 저장 중 오류 발생: {e}")
            raise
            
    def get_data(self,
                symbol: str,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        데이터 조회
        
        Args:
            symbol: 심볼
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            데이터프레임
        """
        try:
            # 데이터 디렉토리
            symbol_dir = os.path.join(self.data_dir, symbol)
            if not os.path.exists(symbol_dir):
                return pd.DataFrame()
                
            # 데이터 파일 목록
            data_files = []
            if start_time and end_time:
                current_date = start_time
                while current_date <= end_time:
                    date_str = current_date.strftime('%Y%m%d')
                    file_path = os.path.join(symbol_dir, f"{date_str}.csv")
                    if os.path.exists(file_path):
                        data_files.append(file_path)
                    current_date += timedelta(days=1)
            else:
                data_files = [os.path.join(symbol_dir, f) for f in os.listdir(symbol_dir)
                            if f.endswith('.csv')]
                
            # 데이터 로드
            dfs = []
            for file_path in data_files:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dfs.append(df)
                
            if not dfs:
                return pd.DataFrame()
                
            # 데이터 병합
            result = pd.concat(dfs, ignore_index=True)
            
            # 시간 필터링
            if start_time:
                result = result[result['timestamp'] >= start_time]
            if end_time:
                result = result[result['timestamp'] <= end_time]
                
            return result.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"데이터 조회 중 오류 발생: {e}")
            raise
            
    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        최신 데이터 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            최신 데이터
        """
        try:
            if symbol in self.data_buffer and self.data_buffer[symbol]:
                return self.data_buffer[symbol][-1]
            return None
            
        except Exception as e:
            self.logger.error(f"최신 데이터 조회 중 오류 발생: {e}")
            raise 