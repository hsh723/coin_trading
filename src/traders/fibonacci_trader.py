import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import ccxt
from ..strategies.fibonacci_strategy import FibonacciStrategy
import logging
import time
import os

class FibonacciTrader:
    def __init__(self, exchange: str, symbol: str, api_key: str, api_secret: str):
        self.exchange = getattr(ccxt, exchange)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        self.symbol = symbol
        self.strategy = FibonacciStrategy()
        self.capital = 1000  # 초기 자본금 (USDT)
        self.current_position = None
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/fibonacci_trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def fetch_1min_data(self) -> pd.DataFrame:
        """1분봉 데이터를 가져옵니다."""
        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol, 
            timeframe='1m', 
            limit=100
        )
        
        df = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    def execute_trade(self, signal: Dict[str, Any]):
        """매매를 실행합니다."""
        if signal['position'] is None:
            return
            
        position_size = self.capital * self.strategy.risk_per_trade
        
        try:
            if signal['position'] == 'long':
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='buy',
                    amount=position_size
                )
                self.logger.info(f"롱 포지션 진입: {order}")
                self.logger.info(f"진입 사유: {signal['reason']}")
                
            elif signal['position'] == 'short':
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='sell',
                    amount=position_size
                )
                self.logger.info(f"숏 포지션 진입: {order}")
                self.logger.info(f"진입 사유: {signal['reason']}")
                
            self.current_position = {
                'type': signal['position'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'size': position_size,
                'reason': signal['reason']
            }
            
        except Exception as e:
            self.logger.error(f"매매 실행 중 오류 발생: {e}")
            
    def check_exit_conditions(self, df: pd.DataFrame) -> bool:
        """청산 조건을 확인합니다."""
        if self.current_position is None:
            return False
            
        current_price = df['close'].iloc[-1]
        
        # 손절/익절 조건 확인
        if self.current_position['type'] == 'long':
            if current_price <= self.current_position['stop_loss']:
                self.logger.info("롱 포지션 손절")
                return True
            elif current_price >= self.current_position['take_profit']:
                self.logger.info("롱 포지션 익절")
                return True
                
        elif self.current_position['type'] == 'short':
            if current_price >= self.current_position['stop_loss']:
                self.logger.info("숏 포지션 손절")
                return True
            elif current_price <= self.current_position['take_profit']:
                self.logger.info("숏 포지션 익절")
                return True
                
        # 피보나치 기반 청산 조건 확인
        should_exit = self.strategy.check_exit_conditions(
            df,
            self.current_position['type'],
            self.current_position['entry_price']
        )
        
        if should_exit:
            self.logger.info(f"{self.current_position['type']} 포지션 피보나치 기반 청산")
            return True
            
        return False
        
    def run(self):
        """트레이딩 루프를 실행합니다."""
        self.logger.info("피보나치 트레이더 시작")
        
        while True:
            try:
                # 1분봉 데이터 가져오기
                df = self.fetch_1min_data()
                
                # 현재 포지션이 없는 경우 신호 생성
                if self.current_position is None:
                    signal = self.strategy.generate_signal(df)
                    self.execute_trade(signal)
                    
                # 현재 포지션이 있는 경우 청산 조건 확인
                else:
                    if self.check_exit_conditions(df):
                        self.current_position = None
                        
                # 1분 대기
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"트레이딩 루프 중 오류 발생: {e}")
                time.sleep(60)  # 오류 발생 시 1분 대기 