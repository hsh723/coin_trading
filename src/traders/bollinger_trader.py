import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import ccxt
from ..strategies.bollinger_strategy import BollingerStrategy
import logging
import time
import os

class BollingerTrader:
    def __init__(self, exchange: str, symbol: str, api_key: str, api_secret: str):
        self.exchange = getattr(ccxt, exchange)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        self.symbol = symbol
        self.strategy = BollingerStrategy()
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
                logging.FileHandler('logs/bollinger_trader.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def fetch_5min_data(self) -> pd.DataFrame:
        """5분봉 데이터를 가져옵니다."""
        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol, 
            timeframe='5m',  # 5분봉 사용
            limit=200  # ABC 패턴을 찾기 위해 더 많은 데이터 필요
        )
        
        df = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
        
    def execute_trade(self, signal: Dict[str, Any]):
        """매매를 실행합니다."""
        if signal['position'] is None:
            return
            
        # 첫 진입은 전체 포지션의 50%만 진입
        position_size = self.capital * self.strategy.risk_per_trade * 0.5
        
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
                'reason': signal['reason'],
                'is_first_entry': True
            }
            
        except Exception as e:
            self.logger.error(f"매매 실행 중 오류 발생: {e}")
            
    def execute_additional_entry(self, df: pd.DataFrame):
        """추가 진입을 실행합니다."""
        if self.current_position is None or not self.current_position['is_first_entry']:
            return
            
        points = self.strategy.find_abc_points(df)
        current_price = df['close'].iloc[-1]
        
        # 다음 저점이 깨진 경우 추가 진입
        if self.current_position['type'] == 'long' and current_price < points['b']:
            additional_size = self.capital * self.strategy.risk_per_trade * 0.5
            
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side='buy',
                    amount=additional_size
                )
                self.logger.info(f"롱 포지션 추가 진입: {order}")
                
                # 포지션 정보 업데이트
                self.current_position['size'] += additional_size
                self.current_position['is_first_entry'] = False
                
            except Exception as e:
                self.logger.error(f"추가 진입 중 오류 발생: {e}")
                
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
                
        # 추세 반전 확인
        df = self.strategy.calculate_indicators(df)
        trend = self.strategy.check_trend(df)
        
        # 롱 포지션에서 하락추세로 전환
        if self.current_position['type'] == 'long' and trend == 'down':
            self.logger.info("롱 포지션 추세 반전 청산")
            return True
            
        # 숏 포지션에서 상승추세로 전환
        elif self.current_position['type'] == 'short' and trend == 'up':
            self.logger.info("숏 포지션 추세 반전 청산")
            return True
            
        return False
        
    def run(self):
        """트레이딩 루프를 실행합니다."""
        self.logger.info("ABC 매매 트레이더 시작")
        
        while True:
            try:
                # 5분봉 데이터 가져오기
                df = self.fetch_5min_data()
                
                # 현재 포지션이 없는 경우 신호 생성
                if self.current_position is None:
                    signal = self.strategy.generate_signal(df)
                    self.execute_trade(signal)
                    
                # 현재 포지션이 있는 경우
                else:
                    # 추가 진입 조건 확인
                    self.execute_additional_entry(df)
                    
                    # 청산 조건 확인
                    if self.check_exit_conditions(df):
                        self.current_position = None
                        
                # 5분 대기
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"트레이딩 루프 중 오류 발생: {e}")
                time.sleep(300)  # 오류 발생 시 5분 대기 