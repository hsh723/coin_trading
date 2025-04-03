import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd
from src.exchange.binance_exchange import BinanceExchange
from ..analysis.news_analyzer import NewsAnalyzer
from ..analysis.technical_analyzer import TechnicalAnalyzer
from ..strategy.integrated_strategy import IntegratedStrategy
from ..risk.risk_manager import RiskManager
from ..database.database import Database
from ..utils.logger import setup_logger
from ..utils.database import DatabaseManager
from ..utils.telegram import TelegramNotifier

class TradingBot(ABC):
    """트레이딩 봇 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        트레이딩 봇 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self.logger = setup_logger('trading_bot')
        self.database = DatabaseManager()
        self.exchange = BinanceExchange(
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret'),
            testnet=config.get('testnet', True)
        )
        self.strategy = IntegratedStrategy(db=self.database)
        self.risk_manager = RiskManager(
            initial_capital=config.get('initial_capital', 10000.0)
        )
        
        # 텔레그램 알림 설정
        bot_token = config.get('telegram_bot_token')
        chat_id = config.get('telegram_chat_id')
        if bot_token and chat_id:
            self.telegram = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        else:
            self.telegram = None
            self.logger.warning("텔레그램 설정이 없습니다. 알림이 비활성화됩니다.")
        
        self.market_data = None
        self.current_position = None
        self._is_running = False
        self.last_update = None
        
    @property
    def is_running(self) -> bool:
        """봇 실행 상태 반환"""
        return self._is_running
        
    @is_running.setter
    def is_running(self, value: bool) -> None:
        """봇 실행 상태 설정"""
        self._is_running = value
        
    async def start(self) -> None:
        """트레이딩 봇 시작"""
        if self.is_running:
            self.logger.warning("트레이딩 봇이 이미 실행 중입니다.")
            return
            
        try:
            self.is_running = True
            self.logger.info("트레이딩 봇 시작")
            self.database.save_log('INFO', '트레이딩 봇 시작', 'trading_bot')
            
            # 메인 루프를 별도의 태스크로 실행
            self._task = asyncio.create_task(self._run_loop())
            await self._task
            
        except Exception as e:
            error_msg = f"트레이딩 봇 시작 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            self.is_running = False
            
    async def stop(self) -> None:
        """트레이딩 봇 중지"""
        if not self.is_running:
            self.logger.warning("트레이딩 봇이 이미 중지되었습니다.")
            return
            
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("트레이딩 봇 중지")
        self.database.save_log('INFO', '트레이딩 봇 중지', 'trading_bot')
        
    async def _run_loop(self) -> None:
        """메인 트레이딩 루프"""
        while self.is_running:
            try:
                # 시장 데이터 업데이트
                await self._update_market_data()
                
                # 거래 가능 여부 확인
                if not self.risk_manager.can_trade():
                    self.logger.warning("거래 불가능 상태")
                    await asyncio.sleep(60)
                    continue
                
                # 거래 신호 생성
                signal = self.strategy.generate_signal(self.market_data)
                
                # 포지션 관리
                if signal and signal['signal'] != 'neutral':
                    await self._manage_position(signal)
                
                # 대기
                await asyncio.sleep(self.config.get('interval', 60))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"트레이딩 루프 오류: {str(e)}"
                self.logger.error(error_msg)
                self.database.save_log('ERROR', error_msg, 'trading_bot')
                await asyncio.sleep(60)
            
    async def _update_market_data(self) -> None:
        """시장 데이터 업데이트"""
        try:
            # OHLCV 데이터 조회
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=self.config['symbol'],
                timeframe=self.config['timeframe'],
                limit=100
            )
            
            if not ohlcv:
                self.logger.warning("OHLCV 데이터 조회 실패")
                return
                
            # DataFrame 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 시장 데이터 업데이트
            self.market_data = {
                'ohlcv': df,
                'current_price': df['close'].iloc[-1],
                'volatility': self.technical_analyzer.calculate_volatility(df),
                'trend_strength': self.technical_analyzer.calculate_trend_strength(df)
            }
            
            # 리스크 파라미터 조정
            self.risk_manager.adjust_risk_parameters(self.market_data['volatility'])
            
            self.logger.info("시장 데이터 업데이트 완료")
            
        except Exception as e:
            error_msg = f"시장 데이터 업데이트 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            
    async def _manage_position(self, signal: Dict[str, Any]) -> None:
        """
        포지션 관리
        
        Args:
            signal (Dict[str, Any]): 거래 신호
        """
        try:
            current_price = self.market_data['current_price']
            
            # 포지션 없음
            if not self.current_position:
                if signal['signal'] == 'buy':
                    # 진입 가격 계산
                    entry_price = current_price
                    
                    # 손절가 계산
                    stop_loss = self.strategy.calculate_stop_loss(
                        entry_price,
                        self.market_data['ohlcv']['atr'].iloc[-1]
                    )
                    
                    # 포지션 크기 계산
                    position_size = self.risk_manager.calculate_position_size(
                        entry_price,
                        stop_loss
                    )
                    
                    # 주문 실행
                    order = self.exchange.create_order(
                        symbol=self.config['symbol'],
                        type='market',
                        side='buy',
                        amount=position_size
                    )
                    
                    if order:
                        self.current_position = {
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': self.strategy.calculate_take_profit(
                                entry_price,
                                stop_loss
                            ),
                            'size': position_size,
                            'order_id': order['id']
                        }
                        
                        # 거래 내역 저장
                        self.database.save_trade({
                            'symbol': self.config['symbol'],
                            'side': 'buy',
                            'price': entry_price,
                            'amount': position_size,
                            'order_id': order['id'],
                            'status': 'open'
                        })
                        
                        # 포지션 정보 저장
                        self.database.save_position({
                            'symbol': self.config['symbol'],
                            'entry_price': entry_price,
                            'amount': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': self.current_position['take_profit'],
                            'status': 'open'
                        })
                        
                        self.logger.info(f"매수 포지션 진입: {self.current_position}")
                        self.database.save_log('INFO', f"매수 포지션 진입: {self.current_position}", 'trading_bot')
                        
            # 포지션 있음
            else:
                # 매도 신호
                if signal['signal'] == 'sell':
                    # 주문 실행
                    order = self.exchange.create_order(
                        symbol=self.config['symbol'],
                        type='market',
                        side='sell',
                        amount=self.current_position['size']
                    )
                    
                    if order:
                        # 손익 계산
                        pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
                        self.risk_manager.update_trade_result(pnl)
                        
                        # 거래 내역 업데이트
                        self.database.save_trade({
                            'symbol': self.config['symbol'],
                            'side': 'sell',
                            'price': current_price,
                            'amount': self.current_position['size'],
                            'order_id': order['id'],
                            'pnl': pnl,
                            'status': 'closed'
                        })
                        
                        # 포지션 정보 업데이트
                        self.database.update_position(
                            self.current_position['id'],
                            {
                                'exit_time': datetime.now(),
                                'exit_price': current_price,
                                'pnl': pnl,
                                'status': 'closed'
                            }
                        )
                        
                        self.logger.info(f"매도 포지션 청산: PNL={pnl:.2f}")
                        self.database.save_log('INFO', f"매도 포지션 청산: PNL={pnl:.2f}", 'trading_bot')
                        self.current_position = None
                        
                # 손절/이익 실현 체크
                elif current_price <= self.current_position['stop_loss']:
                    # 손절 주문 실행
                    order = self.exchange.create_order(
                        symbol=self.config['symbol'],
                        type='market',
                        side='sell',
                        amount=self.current_position['size']
                    )
                    
                    if order:
                        pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
                        self.risk_manager.update_trade_result(pnl)
                        
                        # 거래 내역 업데이트
                        self.database.save_trade({
                            'symbol': self.config['symbol'],
                            'side': 'sell',
                            'price': current_price,
                            'amount': self.current_position['size'],
                            'order_id': order['id'],
                            'pnl': pnl,
                            'status': 'closed'
                        })
                        
                        # 포지션 정보 업데이트
                        self.database.update_position(
                            self.current_position['id'],
                            {
                                'exit_time': datetime.now(),
                                'exit_price': current_price,
                                'pnl': pnl,
                                'status': 'closed'
                            }
                        )
                        
                        self.logger.info(f"손절 실행: PNL={pnl:.2f}")
                        self.database.save_log('INFO', f"손절 실행: PNL={pnl:.2f}", 'trading_bot')
                        self.current_position = None
                        
                elif current_price >= self.current_position['take_profit']:
                    # 이익 실현 주문 실행
                    order = self.exchange.create_order(
                        symbol=self.config['symbol'],
                        type='market',
                        side='sell',
                        amount=self.current_position['size']
                    )
                    
                    if order:
                        pnl = (current_price - self.current_position['entry_price']) * self.current_position['size']
                        self.risk_manager.update_trade_result(pnl)
                        
                        # 거래 내역 업데이트
                        self.database.save_trade({
                            'symbol': self.config['symbol'],
                            'side': 'sell',
                            'price': current_price,
                            'amount': self.current_position['size'],
                            'order_id': order['id'],
                            'pnl': pnl,
                            'status': 'closed'
                        })
                        
                        # 포지션 정보 업데이트
                        self.database.update_position(
                            self.current_position['id'],
                            {
                                'exit_time': datetime.now(),
                                'exit_price': current_price,
                                'pnl': pnl,
                                'status': 'closed'
                            }
                        )
                        
                        self.logger.info(f"이익 실현: PNL={pnl:.2f}")
                        self.database.save_log('INFO', f"이익 실현: PNL={pnl:.2f}", 'trading_bot')
                        self.current_position = None
                        
        except Exception as e:
            error_msg = f"포지션 관리 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            
    def get_status(self) -> Dict[str, Any]:
        """
        봇 상태 조회
        
        Returns:
            Dict[str, Any]: 봇 상태
        """
        try:
            return {
                'is_running': self.is_running,
                'current_position': self.current_position,
                'last_signal': self.last_signal,
                'market_data': {
                    'current_price': self.market_data['current_price'] if self.market_data else None,
                    'volatility': self.market_data['volatility'] if self.market_data else None,
                    'trend_strength': self.market_data['trend_strength'] if self.market_data else None
                },
                'risk_metrics': self.risk_manager.get_risk_metrics()
            }
            
        except Exception as e:
            error_msg = f"상태 조회 실패: {str(e)}"
            self.logger.error(error_msg)
            self.database.save_log('ERROR', error_msg, 'trading_bot')
            return {}
            
    def get_market_data(self) -> Dict[str, Any]:
        """시장 데이터 조회"""
        try:
            if not self.market_data:
                return {}
                
            # OHLCV 데이터를 DataFrame으로 변환
            df = pd.DataFrame(self.market_data['ohlcv'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return {
                'ohlcv': df,
                'current_price': self.market_data['current_price'],
                'volatility': self.market_data['volatility'],
                'trend_strength': self.market_data['trend_strength'],
                'indicators': self.technical_analyzer.calculate_indicators(df)
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 조회 실패: {str(e)}")
            return {}
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """포지션 정보 조회"""
        try:
            if not self.current_position:
                return []
                
            return [{
                'symbol': self.config['symbol'],
                'entry_price': self.current_position['entry_price'],
                'size': self.current_position['size'],
                'stop_loss': self.current_position['stop_loss'],
                'take_profit': self.current_position['take_profit'],
                'pnl': (self.market_data['current_price'] - self.current_position['entry_price']) * self.current_position['size']
            }]
            
        except Exception as e:
            self.logger.error(f"포지션 정보 조회 실패: {str(e)}")
            return []
            
    def get_trades(self) -> List[Dict[str, Any]]:
        """거래 내역 조회"""
        try:
            trades = self.database.get_trades(self.config['symbol'])
            return [{
                'timestamp': trade['timestamp'],
                'symbol': trade['symbol'],
                'side': trade['side'],
                'price': trade['price'],
                'size': trade['size'],
                'pnl': trade.get('pnl', 0.0)
            } for trade in trades]
            
        except Exception as e:
            self.logger.error(f"거래 내역 조회 실패: {str(e)}")
            return [] 