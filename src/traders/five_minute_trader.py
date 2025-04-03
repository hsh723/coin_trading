import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import ccxt
from ..strategies.five_minute_scalping import FiveMinuteScalping
import logging
import time

class FiveMinuteTrader:
    def __init__(self, exchange: str, symbol: str, api_key: str, api_secret: str):
        self.exchange = getattr(ccxt, exchange)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        self.symbol = symbol
        self.strategy = FiveMinuteScalping()
        self.capital = 1000  # 초기 자본금 (USDT)
        self.current_position = None
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/five_minute_trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def fetch_5min_data(self) -> pd.DataFrame:
        """5분봉 데이터를 가져옵니다."""
        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol, 
            timeframe='5m', 
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
            
        position_size = self.strategy.calculate_position_size(
            self.capital, 
            signal['entry_price']
        )
        
        try:
            if signal['position'] == 'long':
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='buy',
                    amount=position_size
                )
                self.logger.info(f"롱 포지션 진입: {order}")
                
            elif signal['position'] == 'short':
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='sell',
                    amount=position_size
                )
                self.logger.info(f"숏 포지션 진입: {order}")
                
            self.current_position = {
                'type': signal['position'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'size': position_size
            }
            
        except Exception as e:
            self.logger.error(f"매매 실행 중 오류 발생: {e}")
            
    def check_exit_conditions(self, current_price: float) -> bool:
        """청산 조건을 확인합니다."""
        if self.current_position is None:
            return False
            
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
                
        return False
        
    def run(self):
        """트레이딩 루프를 실행합니다."""
        self.logger.info("5분봉 스캘핑 트레이더 시작")
        
        while True:
            try:
                # 5분봉 데이터 가져오기
                df = self.fetch_5min_data()
                
                # 현재 포지션이 없는 경우 신호 생성
                if self.current_position is None:
                    signal = self.strategy.generate_signal(df)
                    self.execute_trade(signal)
                    
                # 현재 포지션이 있는 경우 청산 조건 확인
                else:
                    current_price = df['close'].iloc[-1]
                    if self.check_exit_conditions(current_price):
                        self.current_position = None
                        
                # 5분 대기
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"트레이딩 루프 중 오류 발생: {e}")
                time.sleep(60)  # 오류 발생 시 1분 대기 