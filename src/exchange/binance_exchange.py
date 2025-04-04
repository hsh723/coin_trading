import os
from typing import Dict, Any, List, Optional
import ccxt
from src.utils.logger import setup_logger

class BinanceExchange:
    """Binance 거래소 클래스"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        """
        Binance 거래소 초기화
        
        Args:
            api_key (str): API 키
            api_secret (str): API 시크릿
            testnet (bool): 테스트넷 사용 여부
        """
        self.logger = setup_logger('binance_exchange')
        
        # API 키 설정
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API 키가 필요합니다. 환경 변수를 설정하거나 생성자에 전달하세요.")
            
        # 거래소 초기화
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'testnet': testnet
            }
        })
        
        self.logger.info("Binance 거래소 초기화 완료")
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List[float]]:
        """
        OHLCV 데이터 조회
        
        Args:
            symbol (str): 심볼
            timeframe (str): 시간 프레임
            limit (int): 조회할 데이터 수
            
        Returns:
            List[List[float]]: OHLCV 데이터
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            self.logger.info(f"{symbol} OHLCV 데이터 조회 완료")
            return ohlcv
            
        except Exception as e:
            self.logger.error(f"OHLCV 데이터 조회 실패: {str(e)}")
            return []
            
    def fetch_positions(self) -> List[Dict[str, Any]]:
        """
        포지션 정보 조회
        
        Returns:
            List[Dict[str, Any]]: 포지션 정보
        """
        try:
            positions = self.exchange.fetch_positions()
            self.logger.info("포지션 정보 조회 완료")
            return positions
            
        except Exception as e:
            self.logger.error(f"포지션 정보 조회 실패: {str(e)}")
            return []
            
    def fetch_my_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        거래 내역 조회
        
        Args:
            symbol (str): 심볼
            limit (int): 조회할 데이터 수
            
        Returns:
            List[Dict[str, Any]]: 거래 내역
        """
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            self.logger.info(f"{symbol} 거래 내역 조회 완료")
            return trades
            
        except Exception as e:
            self.logger.error(f"거래 내역 조회 실패: {str(e)}")
            return []
            
    def create_order(self, **kwargs) -> Dict[str, Any]:
        """
        주문 생성
        
        Args:
            **kwargs: 주문 파라미터
            
        Returns:
            Dict[str, Any]: 주문 결과
        """
        try:
            order = self.exchange.create_order(**kwargs)
            self.logger.info(f"주문 생성 완료: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"주문 생성 실패: {str(e)}")
            return {}
            
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        주문 취소
        
        Args:
            order_id (str): 주문 ID
            symbol (str): 심볼
            
        Returns:
            Dict[str, Any]: 취소 결과
        """
        try:
            result = self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"주문 취소 완료: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"주문 취소 실패: {str(e)}")
            return {}
            
    def fetch_balance(self) -> Dict[str, Any]:
        """
        잔고 정보 조회
        
        Returns:
            Dict[str, Any]: 잔고 정보
        """
        try:
            balance = self.exchange.fetch_balance()
            self.logger.info("잔고 정보 조회 완료")
            return balance
            
        except Exception as e:
            self.logger.error(f"잔고 정보 조회 실패: {str(e)}")
            return {}
            
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        현재가 정보 조회
        
        Args:
            symbol (str): 심볼
            
        Returns:
            Dict[str, Any]: 현재가 정보
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self.logger.info(f"{symbol} 현재가 정보 조회 완료")
            return ticker
            
        except Exception as e:
            self.logger.error(f"현재가 정보 조회 실패: {str(e)}")
            return {}
            
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        레버리지 설정
        
        Args:
            symbol (str): 심볼
            leverage (int): 레버리지
            
        Returns:
            Dict[str, Any]: 설정 결과
        """
        try:
            result = self.exchange.set_leverage(leverage, symbol)
            self.logger.info(f"{symbol} 레버리지 설정 완료: {leverage}")
            return result
            
        except Exception as e:
            self.logger.error(f"레버리지 설정 실패: {str(e)}")
            return {}
            
    def set_margin_mode(self, symbol: str, mode: str) -> Dict[str, Any]:
        """
        마진 모드 설정
        
        Args:
            symbol (str): 심볼
            mode (str): 마진 모드 (isolated/cross)
            
        Returns:
            Dict[str, Any]: 설정 결과
        """
        try:
            result = self.exchange.set_margin_mode(mode, symbol)
            self.logger.info(f"{symbol} 마진 모드 설정 완료: {mode}")
            return result
            
        except Exception as e:
            self.logger.error(f"마진 모드 설정 실패: {str(e)}")
            return {} 