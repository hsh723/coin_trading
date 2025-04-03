"""
라이브 시뮬레이션 모듈

이 모듈은 실시간 시장 데이터를 기반으로 한 트레이딩 시뮬레이션을 제공합니다.
주요 기능:
- 실시간 시장 데이터 수집 및 처리
- 전략 기반 거래 신호 생성
- 슬리피지 및 부분 체결 시뮬레이션
- 포지션 관리 및 리스크 관리
- 성과 분석 및 보고
- 텔레그램 알림

사용 예시:
```python
# 시뮬레이터 초기화
simulator = LiveSimulator(
    exchange_name='binance',
    initial_capital=10000.0,
    speed=1.0
)

# 시뮬레이터 실행
await simulator.initialize()
await simulator.run_simulation(strategy)
```

설정:
config.yaml 파일에서 다음 설정을 구성할 수 있습니다:
- 시뮬레이션 속도 (speed)
- 슬리피지 (slippage)
- 부분 체결 확률 (partial_fill_probability)
- 시장 변동성 (market_volatility)
- 최대 포지션 수 (max_positions)
- 포지션 크기 제한 (position_size_limit)
- 일일 손실 제한 (max_daily_loss)
- 로깅 레벨 (log_level)
- 텔레그램 알림 설정 (telegram)
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any
from src.trading.simulation import TradingSimulator, SimulationConfig
from src.trading.strategy import IntegratedStrategy
from src.notification.telegram_bot import TelegramNotifier
from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.backtest.visualization import BacktestVisualizer
import os
import json
import time

# 로거 설정
logger = setup_logger()

class LiveSimulator:
    """실시간 시장 데이터 기반 시뮬레이터"""
    
    def __init__(
        self,
        exchange_name: str,
        initial_capital: float,
        speed: float = 1.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.001,    # 0.1%
        results_dir: str = 'data/simulation_results'
    ):
        """
        시뮬레이터 초기화
        
        Args:
            exchange_name (str): 거래소 이름
            initial_capital (float): 초기 자본금
            speed (float): 시뮬레이션 속도
            commission (float): 거래 수수료
            slippage (float): 슬리피지
            results_dir (str): 결과 저장 디렉토리
        """
        self.exchange_name = exchange_name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.speed = speed
        self.commission = commission
        self.slippage = slippage
        self.results_dir = results_dir
        
        # 상태 변수 초기화
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve = pd.Series()
        self.is_running = False
        
        # 시각화기 초기화
        self.visualizer = BacktestVisualizer(results_dir)
        
        # 로거 설정
        self.logger = setup_logger()
        self.logger.info("LiveSimulator initialized")
        
        # 설정 로드
        self.config = get_config()
        
        # 텔레그램 알림 초기화
        if self.config['telegram']['enabled']:
            self.notifier = TelegramNotifier(
                config=self.config,
                bot_token=self.config['telegram']['bot_token'],
                chat_id=self.config['telegram']['chat_id']
            )
        else:
            self.notifier = None
            
        # 시뮬레이션 설정
        sim_config = self.config['simulation']
        config = SimulationConfig(
            initial_capital=initial_capital,
            slippage=sim_config['slippage'],
            partial_fill_probability=sim_config['partial_fill_probability'],
            market_volatility=sim_config['market_volatility'],
            max_positions=sim_config['max_positions'],
            position_size_limit=sim_config['position_size_limit'],
            max_daily_loss=sim_config['max_daily_loss'],
            memory_limit=sim_config['memory_limit']
        )
        self.simulator = TradingSimulator(config)
        
        # 로깅 레벨 설정
        logging.getLogger(__name__).setLevel(sim_config['log_level'])
        
    async def initialize(self):
        """시뮬레이터 초기화"""
        try:
            # 거래소 초기화
            self.exchange = getattr(ccxt, self.exchange_name)({
                'enableRateLimit': True,
                'timeout': self.config['exchange']['timeout']
            })
            
            # 시장 데이터 초기화
            await self._initialize_market_data()
            
            # 전략 초기화
            self.strategy = IntegratedStrategy()
            
            # 시작 메시지 전송
            if self.notifier:
                await self.notifier.send_message(
                    "🚀 시뮬레이션 시작\n"
                    f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"초기 자본: {self.config['trading']['initial_capital']} USDT\n"
                    f"시뮬레이션 속도: {self.speed}x"
                )
            
            logger.info(f"라이브 시뮬레이터 초기화 완료 (속도: {self.speed}x)")
            
        except Exception as e:
            logger.error(f"시뮬레이터 초기화 실패: {str(e)}")
            if self.notifier and self.config['telegram']['notifications']['error']:
                await self.notifier.send_message(f"❌ 시뮬레이터 초기화 실패: {str(e)}")
            raise
            
    async def _initialize_market_data(self):
        """시장 데이터 초기화"""
        try:
            # 설정된 심볼 목록 사용
            symbols = self.config['trading']['symbols']
            timeframe = self.config['trading']['timeframe']
            limit = self.config['trading']['historical_data_limit']
            
            for symbol in symbols:
                # 과거 데이터 로드
                ohlcv = await self._fetch_historical_data(symbol, timeframe, limit)
                self.market_data[symbol] = {
                    'ohlcv': ohlcv,
                    'current_price': float(ohlcv[-1][4]),  # 종가
                    'last_update': datetime.now()
                }
                
            logger.info(f"시장 데이터 초기화 완료 (심볼: {', '.join(symbols)})")
            
        except Exception as e:
            logger.error(f"시장 데이터 초기화 실패: {str(e)}")
            if self.notifier and self.config['telegram']['notifications']['error']:
                await self.notifier.send_message(f"❌ 시장 데이터 초기화 실패: {str(e)}")
            raise
            
    async def _fetch_historical_data(self, symbol: str, timeframe: str, limit: int) -> List:
        """
        과거 데이터 조회
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            limit (int): 데이터 개수
            
        Returns:
            List: OHLCV 데이터
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            return ohlcv
            
        except Exception as e:
            logger.error(f"과거 데이터 조회 실패 ({symbol}): {str(e)}")
            return []
            
    async def run_simulation(self, strategy: IntegratedStrategy):
        """
        시뮬레이션 실행
        
        Args:
            strategy (IntegratedStrategy): 거래 전략
        """
        try:
            self.is_running = True
            self.logger.info(f"시뮬레이션 시작 (속도: {self.speed}x)")
            
            # 시장 데이터 수집
            market_data = await strategy.get_market_data(
                symbol='BTC/USDT',
                timeframe='1h',
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            # 시뮬레이션 루프
            for i in range(len(market_data)):
                if not self.is_running:
                    break
                
                current_data = market_data.iloc[i]
                
                # 포지션 업데이트
                await self._update_positions(current_data)
                
                # 거래 신호 생성
                signal = await strategy.generate_signal(current_data)
                
                if signal:
                    # 거래 실행
                    trade = await self._execute_trade(signal, current_data)
                    if trade:
                        self.trades.append(trade)
                
                # 자본금 곡선 업데이트
                self.equity_curve[current_data.name] = self.current_capital
                
                # 일정 간격으로 대기
                await asyncio.sleep(1 / self.speed)
            
            # 결과 생성 및 저장
            results = self._generate_results()
            report_file = self.visualizer.generate_report(
                results,
                'BTC/USDT',
                '1h'
            )
            
            self.logger.info(f"시뮬레이션 완료: {report_file}")
            
        except Exception as e:
            self.logger.error(f"시뮬레이션 실행 실패: {str(e)}")
            raise
    
    async def _update_positions(self, market_data: pd.Series):
        """
        포지션 상태 업데이트
        
        Args:
            market_data (pd.Series): 시장 데이터
        """
        try:
            for position_id, position in list(self.positions.items()):
                current_price = market_data['close']
                
                # 손절/익절 조건 확인
                if position['side'] == 'buy':
                    pnl = (current_price - position['entry_price']) * position['size']
                    if pnl <= -position['stop_loss']:
                        await self._close_position(position_id, current_price, 'stop_loss')
                    elif pnl >= position['take_profit']:
                        await self._close_position(position_id, current_price, 'take_profit')
                else:
                    pnl = (position['entry_price'] - current_price) * position['size']
                    if pnl <= -position['stop_loss']:
                        await self._close_position(position_id, current_price, 'stop_loss')
                    elif pnl >= position['take_profit']:
                        await self._close_position(position_id, current_price, 'take_profit')
                
                # 포지션 PnL 업데이트
                position['current_price'] = current_price
                position['pnl'] = pnl
                
        except Exception as e:
            self.logger.error(f"포지션 업데이트 실패: {str(e)}")
            raise
    
    async def _execute_trade(self, signal: Dict[str, Any],
                           market_data: pd.Series) -> Optional[Dict[str, Any]]:
        """
        거래 실행
        
        Args:
            signal (Dict[str, Any]): 거래 신호
            market_data (pd.Series): 시장 데이터
            
        Returns:
            Optional[Dict[str, Any]]: 거래 결과
        """
        try:
            # 거래 크기 계산
            position_size = self._calculate_position_size(signal)
            
            # 수수료 계산
            commission = market_data['close'] * position_size * self.commission
            
            # 슬리피지 적용
            if signal['side'] == 'buy':
                entry_price = market_data['close'] * (1 + self.slippage)
            else:
                entry_price = market_data['close'] * (1 - self.slippage)
            
            # 포지션 정보 생성
            position_id = f"{signal['side']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            position = {
                'id': position_id,
                'symbol': signal['symbol'],
                'side': signal['side'],
                'entry_price': entry_price,
                'size': position_size,
                'commission': commission,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_time': market_data.name,
                'current_price': entry_price,
                'pnl': 0.0
            }
            
            # 포지션 진입
            if signal['side'] == 'buy':
                self.current_capital -= (entry_price * position_size + commission)
            else:
                self.current_capital += (entry_price * position_size - commission)
            
            self.positions[position_id] = position
            
            # 거래 기록
            trade = {
                'timestamp': market_data.name,
                'position_id': position_id,
                'symbol': signal['symbol'],
                'side': signal['side'],
                'price': entry_price,
                'size': position_size,
                'commission': commission,
                'type': 'entry'
            }
            
            self.logger.info(
                f"거래 실행: {position_id} - {signal['side']} {position_size} "
                f"@ {entry_price} (수수료: {commission:.2f})"
            )
            
            return trade
            
        except Exception as e:
            self.logger.error(f"거래 실행 실패: {str(e)}")
            return None
    
    async def _close_position(self, position_id: str, current_price: float,
                            reason: str):
        """
        포지션 청산
        
        Args:
            position_id (str): 포지션 ID
            current_price (float): 현재 가격
            reason (str): 청산 이유
        """
        try:
            if position_id not in self.positions:
                raise ValueError(f"포지션을 찾을 수 없음: {position_id}")
            
            position = self.positions[position_id]
            
            # 청산 수수료 계산
            exit_commission = current_price * position['size'] * self.commission
            
            # PnL 계산
            if position['side'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
            
            # 순 PnL 계산 (수수료 제외)
            net_pnl = pnl - position['commission'] - exit_commission
            
            # 자본금 업데이트
            if position['side'] == 'buy':
                self.current_capital += (current_price * position['size'] - exit_commission)
            else:
                self.current_capital -= (current_price * position['size'] + exit_commission)
            
            # 거래 기록
            trade = {
                'timestamp': datetime.now(),
                'position_id': position_id,
                'symbol': position['symbol'],
                'side': 'sell' if position['side'] == 'buy' else 'buy',
                'price': current_price,
                'size': position['size'],
                'commission': exit_commission,
                'pnl': net_pnl,
                'type': 'exit',
                'reason': reason
            }
            self.trades.append(trade)
            
            # 포지션 제거
            del self.positions[position_id]
            
            self.logger.info(
                f"포지션 청산: {position_id} - {position['side']} {position['size']} "
                f"@ {current_price} (PnL: {net_pnl:.2f}, 이유: {reason})"
            )
            
        except Exception as e:
            self.logger.error(f"포지션 청산 실패: {str(e)}")
            raise
    
    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal (Dict[str, Any]): 거래 신호
            
        Returns:
            float: 포지션 크기
        """
        # 리스크 관리 설정
        risk_per_trade = 0.02  # 거래당 2% 리스크
        stop_loss = signal['stop_loss']
        
        # 포지션 크기 계산
        position_size = (self.current_capital * risk_per_trade) / stop_loss
        
        return position_size
    
    def _generate_results(self) -> Dict[str, Any]:
        """
        시뮬레이션 결과 생성
        
        Returns:
            Dict[str, Any]: 시뮬레이션 결과
        """
        try:
            # 수익률 계산
            returns = self.equity_curve.pct_change().dropna()
            
            # 총 수익률
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            
            # 승률
            win_rate = len([t for t in self.trades if t.get('pnl', 0) > 0]) / len(self.trades)
            
            # 최대 낙폭
            rolling_max = self.equity_curve.expanding().max()
            drawdown = (self.equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 샤프 비율
            risk_free_rate = 0.02  # 연간 2% 가정
            excess_returns = returns - risk_free_rate/252  # 일간 무위험 수익률
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 평균 수익률
            avg_return = returns.mean()
            
            # 수익률 표준편차
            return_std = returns.std()
            
            return {
                'total_return': total_return,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(self.trades),
                'avg_return': avg_return,
                'return_std': return_std,
                'equity_curve': self.equity_curve.to_dict(),
                'trades': self.trades
            }
            
        except Exception as e:
            self.logger.error(f"결과 생성 실패: {str(e)}")
            raise
            
    async def close(self):
        """시뮬레이터 종료"""
        self.is_running = False
        if self.exchange:
            await self.exchange.close()
            
        # 종료 메시지 전송
        if self.notifier:
            summary = self.simulator.get_account_summary()
            await self.notifier.send_message(
                "🛑 시뮬레이션 종료\n"
                f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"최종 자산: {summary['total_value']:.2f} USDT\n"
                f"총 수익률: {((summary['total_value'] / self.config['trading']['initial_capital']) - 1) * 100:.2f}%"
            )
            
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """시장 데이터 조회"""
        return self.market_data.get(symbol)
        
    def get_all_market_data(self) -> Dict:
        """전체 시장 데이터 조회"""
        return self.market_data.copy()
        
    def get_simulation_state(self) -> Dict:
        """시뮬레이션 상태 조회"""
        return {
            'is_running': self.is_running,
            'speed': self.speed,
            'market_data': {
                symbol: {
                    'current_price': data['current_price'],
                    'last_update': data['last_update'].isoformat()
                }
                for symbol, data in self.market_data.items()
            },
            'account': self.simulator.get_account_summary()
        } 