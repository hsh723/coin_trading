"""
메인 스크립트
"""

import asyncio
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from src.utils.logger import setup_logger
from src.strategies.integrated import IntegratedStrategy
from src.exchange.binance import BinanceExchange
from src.risk.manager import RiskManager
import signal
from src.utils.api_manager import APIManager
from src.utils.telegram import TelegramNotifier
from src.utils.config import config_manager

# 로거 설정
logger = setup_logger(level='DEBUG')

async def fetch_market_data(
    exchange: BinanceExchange,
    symbol: str,
    timeframe: str = '1h',
    limit: int = 100
) -> pd.DataFrame:
    """
    시장 데이터 조회
    
    Args:
        exchange (BinanceExchange): 거래소 객체
        symbol (str): 심볼
        timeframe (str): 시간 프레임
        limit (int): 데이터 개수
        
    Returns:
        pd.DataFrame: 시장 데이터
    """
    try:
        logger.debug(f"시장 데이터 조회 시작: {symbol}, {timeframe}, {limit}")
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        logger.debug(f"시장 데이터 조회 완료: {len(df)} 개의 데이터")
        return df
        
    except Exception as e:
        logger.error(f"시장 데이터 조회 실패: {str(e)}", exc_info=True)
        return pd.DataFrame()

async def execute_order(
    exchange: BinanceExchange,
    order: Dict[str, Any],
    risk_manager: RiskManager
) -> bool:
    """
    주문 실행
    
    Args:
        exchange (BinanceExchange): 거래소 객체
        order (Dict[str, Any]): 주문 정보
        risk_manager (RiskManager): 리스크 관리자
        
    Returns:
        bool: 실행 성공 여부
    """
    try:
        logger.debug(f"주문 실행 시작: {order}")
        
        # 리스크 체크
        if not risk_manager.check_position_limits():
            logger.warning("포지션 한도 초과")
            return False
            
        if not risk_manager.check_daily_loss_limits():
            logger.warning("일일 손실 한도 초과")
            return False
            
        if not risk_manager.check_drawdown_limits():
            logger.warning("드로다운 한도 초과")
            return False
            
        # 주문 실행
        result = await exchange.create_order(
            symbol=order['symbol'],
            type=order['type'],
            side=order['side'],
            amount=order['amount'],
            price=order['price']
        )
        
        if result:
            # 포지션 정보 업데이트
            risk_manager.add_position({
                'symbol': order['symbol'],
                'side': order['side'],
                'entry_price': order['price'],
                'amount': order['amount'],
                'stop_loss': order['stop_loss'],
                'take_profit': order['take_profit'],
                'timestamp': datetime.now()
            })
            
            logger.info(f"주문 실행 성공: {result}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"주문 실행 실패: {str(e)}", exc_info=True)
        return False

async def close_position(
    exchange: BinanceExchange,
    position: Dict[str, Any],
    risk_manager: RiskManager
) -> bool:
    """
    포지션 종료
    
    Args:
        exchange (BinanceExchange): 거래소 객체
        position (Dict[str, Any]): 포지션 정보
        risk_manager (RiskManager): 리스크 관리자
        
    Returns:
        bool: 종료 성공 여부
    """
    try:
        logger.debug(f"포지션 종료 시작: {position}")
        
        # 반대 방향 주문
        close_side = 'sell' if position['side'] == 'buy' else 'buy'
        
        result = await exchange.create_order(
            symbol=position['symbol'],
            type='market',
            side=close_side,
            amount=position['amount']
        )
        
        if result:
            # 포지션 제거
            risk_manager.remove_position(position['symbol'])
            
            # 손익 계산
            pnl = (result['price'] - position['entry_price']) * position['amount']
            if position['side'] == 'sell':
                pnl = -pnl
                
            # 일일 손익 업데이트
            risk_manager.update_daily_pnl(pnl)
            
            logger.info(f"포지션 종료 성공: {result}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"포지션 종료 실패: {str(e)}", exc_info=True)
        return False

class TradingBot:
    def __init__(self):
        """트레이딩 봇 초기화"""
        self.api_manager = APIManager()
        self.telegram = TelegramNotifier()
        self.is_running = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """종료 시그널 처리"""
        logger.info(f"종료 시그널 수신: {signum}")
        self.stop()
    
    def start(self):
        """트레이딩 봇 시작"""
        try:
            logger.info("트레이딩 봇 시작")
            self.telegram.send_info("트레이딩 봇이 시작되었습니다.")
            
            self.is_running = True
            while self.is_running:
                try:
                    # 트레이딩 로직 실행
                    self._run_trading_cycle()
                    
                except Exception as e:
                    logger.error(f"트레이딩 사이클 오류: {str(e)}")
                    self.telegram.send_error(e, "트레이딩 사이클")
                    
                    # 심각한 오류 발생 시 안전하게 종료
                    if self._is_critical_error(e):
                        self.stop()
                        break
                    
                    # 일시적인 오류는 재시도
                    continue
            
        except Exception as e:
            logger.error(f"트레이딩 봇 실행 중 오류: {str(e)}")
            self.telegram.send_error(e, "트레이딩 봇 실행")
            self.stop()
    
    def stop(self):
        """트레이딩 봇 종료"""
        if not self.is_running:
            return
        
        logger.info("트레이딩 봇 종료 중...")
        self.telegram.send_info("트레이딩 봇이 종료됩니다.")
        
        try:
            # 모든 포지션 정리
            self.api_manager.close_all_positions()
            
            # 리소스 정리
            self.is_running = False
            
            logger.info("트레이딩 봇 종료 완료")
            self.telegram.send_info("트레이딩 봇이 정상적으로 종료되었습니다.")
            
        except Exception as e:
            logger.error(f"트레이딩 봇 종료 중 오류: {str(e)}")
            self.telegram.send_error(e, "트레이딩 봇 종료")
            sys.exit(1)
    
    def _run_trading_cycle(self):
        """트레이딩 사이클 실행"""
        # 여기에 실제 트레이딩 로직 구현
        pass
    
    def _is_critical_error(self, error: Exception) -> bool:
        """심각한 오류 여부 확인"""
        critical_errors = [
            'ConnectionError',
            'AuthenticationError',
            'InsufficientFunds',
            'InvalidOrder',
            'ExchangeError'
        ]
        
        error_type = type(error).__name__
        return error_type in critical_errors

def main():
    """메인 함수"""
    try:
        bot = TradingBot()
        bot.start()
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Windows에서 SelectorEventLoop 사용
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main() 