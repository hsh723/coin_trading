"""
API 연결 관리 모듈
"""

import ccxt
import time
from datetime import datetime
import logging
from typing import Optional, Dict, Any
from .config import config_manager
from .logger import logger
from .telegram import TelegramNotifier

class APIManager:
    def __init__(self, exchange_id: str = 'binance'):
        """API 관리자 초기화"""
        self.exchange_id = exchange_id
        self.exchange: Optional[ccxt.Exchange] = None
        self.telegram = TelegramNotifier()
        self.max_retries = 3
        self.retry_delay = 5  # 초
        self.last_connection_time = None
        self.connection_timeout = 30  # 초
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """거래소 초기화"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': config_manager.get_config('BINANCE_API_KEY'),
                'secret': config_manager.get_config('BINANCE_API_SECRET'),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            logger.info(f"{self.exchange_id} 거래소 초기화 완료")
        except Exception as e:
            logger.error(f"거래소 초기화 실패: {str(e)}")
            self._notify_error(f"거래소 초기화 실패: {str(e)}")
    
    def _check_connection(self) -> bool:
        """연결 상태 확인"""
        if not self.exchange:
            return False
        
        try:
            # 간단한 API 호출로 연결 상태 확인
            self.exchange.fetch_time()
            self.last_connection_time = datetime.now()
            return True
        except Exception as e:
            logger.warning(f"연결 상태 확인 실패: {str(e)}")
            return False
    
    def _reconnect(self) -> bool:
        """재연결 시도"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"재연결 시도 {attempt + 1}/{self.max_retries}")
                self._initialize_exchange()
                if self._check_connection():
                    logger.info("재연결 성공")
                    self._notify_recovery("API 연결이 복구되었습니다.")
                    return True
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"재연결 시도 실패: {str(e)}")
                time.sleep(self.retry_delay)
        
        logger.error("최대 재연결 시도 횟수 초과")
        self._notify_error("API 연결 복구 실패. 프로그램을 종료합니다.")
        return False
    
    def safe_api_call(self, func, *args, **kwargs) -> Any:
        """
        안전한 API 호출
        
        Args:
            func: 호출할 API 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
            
        Returns:
            API 호출 결과
        """
        for attempt in range(self.max_retries):
            try:
                # 연결 상태 확인
                if not self._check_connection():
                    if not self._reconnect():
                        raise ConnectionError("API 연결 복구 실패")
                
                # API 호출
                result = func(*args, **kwargs)
                return result
                
            except ccxt.NetworkError as e:
                logger.error(f"네트워크 오류: {str(e)}")
                self._notify_error(f"네트워크 오류 발생: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
                    
            except ccxt.ExchangeError as e:
                logger.error(f"거래소 오류: {str(e)}")
                self._notify_error(f"거래소 오류 발생: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"예상치 못한 오류: {str(e)}")
                self._notify_error(f"예상치 못한 오류 발생: {str(e)}")
                raise
    
    def close_all_positions(self):
        """모든 포지션 정리"""
        try:
            if not self.exchange:
                raise ConnectionError("거래소 연결이 없습니다.")
            
            # 현재 포지션 조회
            positions = self.safe_api_call(
                self.exchange.fetch_positions
            )
            
            for position in positions:
                if float(position['contracts']) > 0:
                    # 포지션 종료
                    self.safe_api_call(
                        self.exchange.create_order,
                        symbol=position['symbol'],
                        type='market',
                        side='sell',
                        amount=position['contracts']
                    )
                    logger.info(f"포지션 종료: {position['symbol']}")
            
            logger.info("모든 포지션이 정리되었습니다.")
            self._notify_info("모든 포지션이 안전하게 정리되었습니다.")
            
        except Exception as e:
            logger.error(f"포지션 정리 실패: {str(e)}")
            self._notify_error(f"포지션 정리 실패: {str(e)}")
            raise
    
    def _notify_error(self, message: str):
        """오류 알림"""
        self.telegram.send_message(f"❌ 오류 발생: {message}")
    
    def _notify_recovery(self, message: str):
        """복구 알림"""
        self.telegram.send_message(f"✅ 복구 완료: {message}")
    
    def _notify_info(self, message: str):
        """정보 알림"""
        self.telegram.send_message(f"ℹ️ 정보: {message}")
    
    def __del__(self):
        """소멸자"""
        if self.exchange:
            try:
                self.exchange.close()
            except:
                pass 