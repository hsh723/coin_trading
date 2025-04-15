import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
import websockets
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
import numpy as np
from datetime import datetime
import os
from prometheus_client import Counter, Histogram, Gauge
import time

# 로깅 설정
logger = logging.getLogger(__name__)

class RealTimeStreaming:
    """
    실시간 데이터 스트리밍 시스템
    
    주요 기능:
    - WebSocket 기반 실시간 데이터 수집
    - Kafka 기반 데이터 파이프라인
    - 실시간 데이터 전처리
    - 데이터 품질 모니터링
    - 이상치 탐지 통합
    """
    
    def __init__(self,
                 websocket_url: str,
                 kafka_bootstrap_servers: List[str],
                 topics: List[str],
                 batch_size: int = 100,
                 batch_timeout: float = 1.0,
                 save_dir: str = "./streaming_data"):
        """
        스트리밍 시스템 초기화
        
        Args:
            websocket_url: WebSocket 서버 URL
            kafka_bootstrap_servers: Kafka 브로커 주소 리스트
            topics: 구독할 토픽 리스트
            batch_size: 배치 처리 크기
            batch_timeout: 배치 처리 타임아웃 (초)
            save_dir: 데이터 저장 디렉토리
        """
        self.websocket_url = websocket_url
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.topics = topics
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.save_dir = save_dir
        
        # 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # Kafka 프로듀서/컨슈머 초기화
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # 메트릭스 초기화
        self.messages_received = Counter('messages_received_total', '수신된 메시지 수')
        self.messages_processed = Counter('messages_processed_total', '처리된 메시지 수')
        self.processing_latency = Histogram('processing_latency_seconds', '처리 지연 시간')
        self.buffer_size = Gauge('buffer_size', '버퍼 크기')
        
        # 데이터 버퍼
        self.data_buffer = []
        self.last_batch_time = time.time()
        
        # 실행 상태
        self.is_running = False
        
    async def start(self):
        """스트리밍 시스템 시작"""
        try:
            self.is_running = True
            logger.info("스트리밍 시스템 시작")
            
            # WebSocket 연결
            async with websockets.connect(self.websocket_url) as websocket:
                while self.is_running:
                    try:
                        # 메시지 수신
                        message = await websocket.recv()
                        self.messages_received.inc()
                        
                        # 데이터 파싱
                        data = json.loads(message)
                        
                        # 데이터 전처리
                        processed_data = self._preprocess_data(data)
                        
                        # Kafka로 전송
                        self.producer.send('market_data', processed_data)
                        
                        # 버퍼에 추가
                        self.data_buffer.append(processed_data)
                        self.buffer_size.set(len(self.data_buffer))
                        
                        # 배치 처리
                        if (len(self.data_buffer) >= self.batch_size or 
                            time.time() - self.last_batch_time >= self.batch_timeout):
                            await self._process_batch()
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket 연결 종료, 재연결 시도 중...")
                        break
                    except Exception as e:
                        logger.error(f"메시지 처리 중 오류 발생: {e}")
                        
        except Exception as e:
            logger.error(f"스트리밍 시스템 실행 중 오류 발생: {e}")
        finally:
            self.is_running = False
            
    async def _process_batch(self):
        """배치 데이터 처리"""
        try:
            start_time = time.time()
            
            if not self.data_buffer:
                return
                
            # 데이터프레임 생성
            df = pd.DataFrame(self.data_buffer)
            
            # 이상치 탐지
            anomalies = self._detect_anomalies(df)
            
            # 데이터 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.save_dir, f'batch_{timestamp}.csv')
            df.to_csv(filename, index=False)
            
            # 메트릭스 업데이트
            self.messages_processed.inc(len(self.data_buffer))
            self.processing_latency.observe(time.time() - start_time)
            
            # 버퍼 초기화
            self.data_buffer = []
            self.last_batch_time = time.time()
            
            logger.info(f"배치 처리 완료: {len(df)}개 레코드, {len(anomalies)}개 이상치")
            
        except Exception as e:
            logger.error(f"배치 처리 중 오류 발생: {e}")
            
    def _preprocess_data(self, data: Dict) -> Dict:
        """
        데이터 전처리
        
        Args:
            data: 원본 데이터
            
        Returns:
            전처리된 데이터
        """
        try:
            # 필수 필드 확인
            required_fields = ['timestamp', 'symbol', 'price', 'volume']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"필수 필드 누락: {field}")
                    
            # 타임스탬프 변환
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # 수치형 데이터 변환
            data['price'] = float(data['price'])
            data['volume'] = float(data['volume'])
            
            # 추가 계산 필드
            data['price_change'] = 0.0  # 이전 가격과의 차이
            data['volume_change'] = 0.0  # 이전 거래량과의 차이
            
            return data
            
        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {e}")
            return {}
            
    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 탐지
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            이상치 데이터프레임
        """
        try:
            # 가격 변동성 기반 이상치 탐지
            price_std = df['price'].std()
            price_mean = df['price'].mean()
            price_threshold = 3 * price_std
            
            # 거래량 기반 이상치 탐지
            volume_std = df['volume'].std()
            volume_mean = df['volume'].mean()
            volume_threshold = 3 * volume_std
            
            # 이상치 마스크
            price_anomalies = (df['price'] - price_mean).abs() > price_threshold
            volume_anomalies = (df['volume'] - volume_mean).abs() > volume_threshold
            
            # 이상치 데이터프레임
            anomalies = df[price_anomalies | volume_anomalies].copy()
            anomalies['anomaly_type'] = np.where(price_anomalies, 'price', 'volume')
            
            return anomalies
            
        except Exception as e:
            logger.error(f"이상치 탐지 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def stop(self):
        """스트리밍 시스템 종료"""
        self.is_running = False
        self.producer.close()
        self.consumer.close()
        logger.info("스트리밍 시스템 종료") 