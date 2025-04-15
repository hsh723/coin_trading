import asyncio
import logging
import websockets
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import time

logger = logging.getLogger(__name__)

class DataCollector:
    """
    실시간 데이터 수집 시스템
    
    주요 기능:
    - WebSocket을 통한 실시간 데이터 수집
    - REST API를 통한 과거 데이터 수집
    - 데이터 전처리 및 저장
    - 데이터 품질 모니터링
    """
    
    def __init__(self,
                 websocket_url: str,
                 rest_api_url: str,
                 symbols: List[str],
                 save_dir: str = "./data"):
        """
        데이터 수집기 초기화
        
        Args:
            websocket_url: WebSocket 서버 URL
            rest_api_url: REST API 서버 URL
            symbols: 수집할 심볼 목록
            save_dir: 데이터 저장 디렉토리
        """
        self.websocket_url = websocket_url
        self.rest_api_url = rest_api_url
        self.symbols = symbols
        self.save_dir = save_dir
        
        # 데이터 버퍼
        self.data_buffer = {symbol: [] for symbol in symbols}
        self.buffer_size = 1000
        
        # 연결 상태
        self.connected = False
        self.ws = None
        
        # 메트릭스
        self.metrics = {
            'messages_received': 0,
            'messages_processed': 0,
            'processing_time': 0,
            'errors': 0
        }
    
    async def start(self):
        """데이터 수집 시작"""
        logger.info("데이터 수집 시작")
        self.connected = True
        
        # WebSocket 연결
        while self.connected:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.ws = websocket
                    logger.info("WebSocket 연결 성공")
                    
                    # 구독 메시지 전송
                    subscribe_message = {
                        "method": "SUBSCRIBE",
                        "params": [f"{symbol.lower()}@trade" for symbol in self.symbols],
                        "id": 1
                    }
                    await websocket.send(json.dumps(subscribe_message))
                    
                    # 데이터 수신 루프
                    while self.connected:
                        try:
                            message = await websocket.recv()
                            await self._process_message(message)
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket 연결 종료")
                            break
                        except Exception as e:
                            logger.error(f"메시지 처리 중 오류: {e}")
                            self.metrics['errors'] += 1
                            
            except Exception as e:
                logger.error(f"WebSocket 연결 중 오류: {e}")
                self.metrics['errors'] += 1
                await asyncio.sleep(5)  # 재연결 전 대기
    
    async def stop(self):
        """데이터 수집 중지"""
        logger.info("데이터 수집 중지")
        self.connected = False
        if self.ws:
            await self.ws.close()
    
    async def _process_message(self, message: str):
        """메시지 처리"""
        start_time = time.time()
        
        try:
            data = json.loads(message)
            self.metrics['messages_received'] += 1
            
            # 거래 데이터 처리
            if 's' in data and 'p' in data and 'q' in data:
                symbol = data['s']
                price = float(data['p'])
                quantity = float(data['q'])
                timestamp = datetime.fromtimestamp(data['T'] / 1000)
                
                # 데이터 버퍼에 추가
                self.data_buffer[symbol].append({
                    'timestamp': timestamp,
                    'price': price,
                    'quantity': quantity
                })
                
                # 버퍼 크기 제한
                if len(self.data_buffer[symbol]) > self.buffer_size:
                    self.data_buffer[symbol].pop(0)
                
                # 데이터 저장
                if len(self.data_buffer[symbol]) % 100 == 0:
                    await self._save_data(symbol)
            
            self.metrics['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"메시지 처리 중 오류: {e}")
            self.metrics['errors'] += 1
        
        self.metrics['processing_time'] += time.time() - start_time
    
    async def _save_data(self, symbol: str):
        """데이터 저장"""
        try:
            # 데이터프레임 생성
            df = pd.DataFrame(self.data_buffer[symbol])
            
            # 파일명 생성
            filename = f"{self.save_dir}/{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # 데이터 저장
            df.to_csv(filename, index=False)
            logger.info(f"데이터 저장 완료: {filename}")
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류: {e}")
            self.metrics['errors'] += 1
    
    async def get_historical_data(self,
                                symbol: str,
                                interval: str = '1m',
                                start_time: Optional[int] = None,
                                end_time: Optional[int] = None) -> pd.DataFrame:
        """과거 데이터 수집"""
        try:
            params = {
                'symbol': symbol,
                'interval': interval
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.rest_api_url}/klines", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        return df
                    else:
                        logger.error(f"과거 데이터 수집 실패: {response.status}")
                        return pd.DataFrame()
                        
        except Exception as e:
            logger.error(f"과거 데이터 수집 중 오류: {e}")
            return pd.DataFrame()
    
    def get_metrics(self) -> Dict[str, Any]:
        """수집 메트릭스 반환"""
        return self.metrics 