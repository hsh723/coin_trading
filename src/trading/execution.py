"""
주문 실행 모듈
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.utils.logger import setup_logger
from src.utils.config_loader import get_config

logger = setup_logger('trading_execution')

class OrderExecutor:
    def __init__(
        self,
        exchange: str,
        testnet: bool = False,
        initial_capital: float = 10000.0
    ):
        """
        주문 실행기 초기화
        
        Args:
            exchange (str): 거래소 이름
            testnet (bool): 테스트넷 사용 여부
            initial_capital (float): 초기 자본금
        """
        self.exchange = exchange
        self.testnet = testnet
        self.initial_capital = initial_capital
        self.logger = setup_logger()
        
        # 설정 로드
        self.config = get_config()
        
        # API 설정
        self.api_key = self.config['exchange']['api_key']
        self.api_secret = self.config['exchange']['api_secret']
        
        # 세션 관리
        self.session = None
        self.ws_session = None
        
        # 상태 변수
        self.is_connected = False
        self.last_order_id = None
        self.order_book = {}
        self.trades = []
        
    async def initialize(self):
        """초기화"""
        try:
            # HTTP 세션 생성
            self.session = aiohttp.ClientSession()
            
            # WebSocket 연결
            await self._connect_websocket()
            
            # 거래소 초기화
            await self._initialize_exchange()
            
            self.logger.info(f"{self.exchange} 거래소 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"거래소 초기화 실패: {str(e)}")
            raise
            
    async def close(self):
        """리소스 정리"""
        try:
            if self.session:
                await self.session.close()
            if self.ws_session:
                await self.ws_session.close()
                
            self.logger.info("거래소 연결 종료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {str(e)}")
            raise
            
    async def _connect_websocket(self):
        """WebSocket 연결"""
        try:
            # WebSocket URL 설정
            ws_url = self.config['exchange']['websocket_url']
            if self.testnet:
                ws_url = self.config['exchange']['testnet_websocket_url']
                
            # WebSocket 연결
            self.ws_session = await self.session.ws_connect(ws_url)
            
            # 구독 메시지 전송
            await self._subscribe_websocket()
            
            self.is_connected = True
            self.logger.info("WebSocket 연결 완료")
            
        except Exception as e:
            self.logger.error(f"WebSocket 연결 실패: {str(e)}")
            raise
            
    async def _subscribe_websocket(self):
        """WebSocket 구독"""
        try:
            # 구독할 채널 설정
            channels = [
                'ticker',
                'orderbook',
                'trades',
                'user_trades'
            ]
            
            # 구독 메시지 전송
            subscribe_msg = {
                'method': 'SUBSCRIBE',
                'params': channels,
                'id': 1
            }
            
            await self.ws_session.send_json(subscribe_msg)
            
            # 응답 대기
            response = await self.ws_session.receive_json()
            if response.get('result') is None:
                raise Exception(f"구독 실패: {response}")
                
            self.logger.info("WebSocket 구독 완료")
            
        except Exception as e:
            self.logger.error(f"WebSocket 구독 실패: {str(e)}")
            raise
            
    async def _initialize_exchange(self):
        """거래소 초기화"""
        try:
            # 계정 정보 조회
            account_info = await self._get_account_info()
            
            # 거래 가능한 심볼 조회
            symbols = await self._get_trading_symbols()
            
            # 주문서 초기화
            for symbol in symbols:
                self.order_book[symbol] = {
                    'bids': [],
                    'asks': []
                }
                
            self.logger.info(f"거래소 초기화 완료: {len(symbols)}개 심볼")
            
        except Exception as e:
            self.logger.error(f"거래소 초기화 실패: {str(e)}")
            raise
            
    async def _get_account_info(self) -> Dict[str, Any]:
        """
        계정 정보 조회
        
        Returns:
            Dict[str, Any]: 계정 정보
        """
        try:
            # API 엔드포인트 설정
            endpoint = '/api/v3/account'
            
            # 요청 헤더 설정
            headers = self._get_auth_headers('GET', endpoint)
            
            # API 요청
            async with self.session.get(
                f"{self.config['exchange']['api_url']}{endpoint}",
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"계정 정보 조회 실패: {response.text}")
                    
                return await response.json()
                
        except Exception as e:
            self.logger.error(f"계정 정보 조회 실패: {str(e)}")
            raise
            
    async def _get_trading_symbols(self) -> List[str]:
        """
        거래 가능한 심볼 조회
        
        Returns:
            List[str]: 심볼 목록
        """
        try:
            # API 엔드포인트 설정
            endpoint = '/api/v3/exchangeInfo'
            
            # API 요청
            async with self.session.get(
                f"{self.config['exchange']['api_url']}{endpoint}"
            ) as response:
                if response.status != 200:
                    raise Exception(f"심볼 정보 조회 실패: {response.text}")
                    
                data = await response.json()
                return [
                    symbol['symbol']
                    for symbol in data['symbols']
                    if symbol['status'] == 'TRADING'
                ]
                
        except Exception as e:
            self.logger.error(f"거래 가능한 심볼 조회 실패: {str(e)}")
            raise
            
    async def get_market_data(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            symbol (str): 거래 심볼
            interval (str): 시간 간격
            limit (int): 데이터 개수
            
        Returns:
            pd.DataFrame: 시장 데이터
        """
        try:
            # API 엔드포인트 설정
            endpoint = '/api/v3/klines'
            
            # 요청 파라미터 설정
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            # API 요청
            async with self.session.get(
                f"{self.config['exchange']['api_url']}{endpoint}",
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(f"시장 데이터 조회 실패: {response.text}")
                    
                data = await response.json()
                
                # DataFrame 생성
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # 데이터 타입 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                    
                return df
                
        except Exception as e:
            self.logger.error(f"시장 데이터 조회 실패: {str(e)}")
            raise
            
    async def execute_order(
        self,
        signal: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        주문 실행
        
        Args:
            signal (Dict[str, Any]): 거래 신호
            
        Returns:
            Optional[Dict[str, Any]]: 주문 결과
        """
        try:
            # 주문 파라미터 설정
            params = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'type': signal['type'],
                'quantity': signal['quantity']
            }
            
            if signal['type'] == 'LIMIT':
                params['price'] = signal['price']
                params['timeInForce'] = 'GTC'
                
            # API 엔드포인트 설정
            endpoint = '/api/v3/order'
            
            # 요청 헤더 설정
            headers = self._get_auth_headers('POST', endpoint, params)
            
            # API 요청
            async with self.session.post(
                f"{self.config['exchange']['api_url']}{endpoint}",
                headers=headers,
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(f"주문 실행 실패: {response.text}")
                    
                order = await response.json()
                
                # 주문 정보 저장
                self.last_order_id = order['orderId']
                self.trades.append(order)
                
                return order
                
        except Exception as e:
            self.logger.error(f"주문 실행 실패: {str(e)}")
            raise
            
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        포지션 조회
        
        Returns:
            List[Dict[str, Any]]: 포지션 목록
        """
        try:
            # API 엔드포인트 설정
            endpoint = '/api/v3/positionRisk'
            
            # 요청 헤더 설정
            headers = self._get_auth_headers('GET', endpoint)
            
            # API 요청
            async with self.session.get(
                f"{self.config['exchange']['api_url']}{endpoint}",
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"포지션 조회 실패: {response.text}")
                    
                positions = await response.json()
                
                # 0이 아닌 포지션만 필터링
                return [
                    position
                    for position in positions
                    if float(position['positionAmt']) != 0
                ]
                
        except Exception as e:
            self.logger.error(f"포지션 조회 실패: {str(e)}")
            raise
            
    async def calculate_pnl(
        self,
        position: Dict[str, Any]
    ) -> float:
        """
        손익 계산
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            
        Returns:
            float: 손익
        """
        try:
            # 현재가 조회
            current_price = await self._get_current_price(position['symbol'])
            
            # 진입가와 현재가의 차이 계산
            entry_price = float(position['entryPrice'])
            position_amt = float(position['positionAmt'])
            
            if position_amt > 0:  # 롱 포지션
                pnl = (current_price - entry_price) * position_amt
            else:  # 숏 포지션
                pnl = (entry_price - current_price) * abs(position_amt)
                
            return pnl
            
        except Exception as e:
            self.logger.error(f"손익 계산 실패: {str(e)}")
            raise
            
    async def close_position(
        self,
        position: Dict[str, Any],
        reason: str = 'manual'
    ) -> Optional[Dict[str, Any]]:
        """
        포지션 종료
        
        Args:
            position (Dict[str, Any]): 포지션 정보
            reason (str): 종료 사유
            
        Returns:
            Optional[Dict[str, Any]]: 종료 결과
        """
        try:
            # 반대 방향으로 청산 주문
            close_side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
            
            # 주문 파라미터 설정
            params = {
                'symbol': position['symbol'],
                'side': close_side,
                'type': 'MARKET',
                'quantity': abs(float(position['positionAmt']))
            }
            
            # API 엔드포인트 설정
            endpoint = '/api/v3/order'
            
            # 요청 헤더 설정
            headers = self._get_auth_headers('POST', endpoint, params)
            
            # API 요청
            async with self.session.post(
                f"{self.config['exchange']['api_url']}{endpoint}",
                headers=headers,
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(f"포지션 종료 실패: {response.text}")
                    
                order = await response.json()
                
                # 거래 기록에 종료 사유 추가
                order['close_reason'] = reason
                self.trades.append(order)
                
                return order
                
        except Exception as e:
            self.logger.error(f"포지션 종료 실패: {str(e)}")
            raise
            
    async def _get_current_price(self, symbol: str) -> float:
        """
        현재가 조회
        
        Args:
            symbol (str): 거래 심볼
            
        Returns:
            float: 현재가
        """
        try:
            # API 엔드포인트 설정
            endpoint = '/api/v3/ticker/price'
            
            # 요청 파라미터 설정
            params = {'symbol': symbol}
            
            # API 요청
            async with self.session.get(
                f"{self.config['exchange']['api_url']}{endpoint}",
                params=params
            ) as response:
                if response.status != 200:
                    raise Exception(f"현재가 조회 실패: {response.text}")
                    
                data = await response.json()
                return float(data['price'])
                
        except Exception as e:
            self.logger.error(f"현재가 조회 실패: {str(e)}")
            raise
            
    def _get_auth_headers(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        인증 헤더 생성
        
        Args:
            method (str): HTTP 메서드
            endpoint (str): API 엔드포인트
            params (Dict[str, Any]): 요청 파라미터
            
        Returns:
            Dict[str, str]: 인증 헤더
        """
        try:
            # 타임스탬프 생성
            timestamp = int(datetime.now().timestamp() * 1000)
            
            # 쿼리 문자열 생성
            query_string = ''
            if params:
                query_string = '&'.join(
                    f"{k}={v}" for k, v in sorted(params.items())
                )
                
            # 서명 생성
            signature_payload = f"{timestamp}{method}{endpoint}{query_string}"
            signature = self._generate_signature(signature_payload)
            
            # 헤더 생성
            return {
                'X-MBX-APIKEY': self.api_key,
                'X-MBX-TIMESTAMP': str(timestamp),
                'X-MBX-SIGNATURE': signature
            }
            
        except Exception as e:
            self.logger.error(f"인증 헤더 생성 실패: {str(e)}")
            raise
            
    def _generate_signature(self, payload: str) -> str:
        """
        서명 생성
        
        Args:
            payload (str): 서명할 데이터
            
        Returns:
            str: 서명
        """
        try:
            import hmac
            import hashlib
            
            return hmac.new(
                self.api_secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
        except Exception as e:
            self.logger.error(f"서명 생성 실패: {str(e)}")
            raise 