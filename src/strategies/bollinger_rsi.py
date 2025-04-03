import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from .base import Strategy
from ..utils.config_loader import get_config

class BollingerRSIStrategy(Strategy):
    def __init__(self, 
                 name: str,
                 symbol: str,
                 timeframe: str,
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.04):
        """
        볼린저 밴드 + RSI 전략 초기화
        
        Args:
            name: 전략 이름
            symbol: 거래 페어
            timeframe: 시간 프레임
            initial_capital: 초기 자본금
            position_size: 포지션 크기 (자본금 대비)
            stop_loss: 손절 비율
            take_profit: 익절 비율
        """
        super().__init__(name, symbol, timeframe, initial_capital, position_size, stop_loss, take_profit)
        
        # 전략 파라미터 로드
        self.params = get_config()['trading_params']['bollinger_rsi']
        
        # 지표 파라미터
        self.bb_period = self.params['bollinger_bands']['period']
        self.bb_std = self.params['bollinger_bands']['std_dev']
        self.rsi_period = self.params['rsi']['period']
        self.rsi_oversold = self.params['rsi']['oversold']
        self.rsi_overbought = self.params['rsi']['overbought']
        
        # 신호 파라미터
        self.volume_threshold = self.params['signals']['volume_threshold']
        self.trend_period = self.params['signals']['trend_period']
        
        self.logger.info(f"볼린저 밴드 + RSI 전략 초기화 완료: {name}")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        전략 초기화
        
        Args:
            data: OHLCV 데이터프레임
        """
        self.logger.info("전략 초기화 시작")
        
        # 데이터 전처리
        data['returns'] = data['close'].pct_change()
        data['volume_ma'] = data['volume'].rolling(window=self.trend_period).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        self.logger.info("전략 초기화 완료")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        기술 지표 계산
        
        Args:
            data: OHLCV 데이터프레임
            
        Returns:
            DataFrame: 지표가 추가된 데이터프레임
        """
        try:
            self.logger.info("기술 지표 계산 시작")
            
            # 볼린저 밴드 계산
            data = self.indicators.calculate_bollinger_bands(
                data,
                period=self.bb_period,
                std_dev=self.bb_std
            )
            
            # RSI 계산
            data = self.indicators.calculate_rsi(
                data,
                period=self.rsi_period
            )
            
            # 추세 방향 계산
            data['trend'] = np.where(
                data['close'] > data[f'sma_{self.trend_period}'],
                1,  # 상승 추세
                -1  # 하락 추세
            )
            
            self.logger.info("기술 지표 계산 완료")
            return data
            
        except Exception as e:
            self.logger.error(f"기술 지표 계산 중 오류 발생: {str(e)}")
            raise
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        매매 신호 생성
        
        Args:
            data: OHLCV 데이터프레임
            
        Returns:
            DataFrame: 신호가 추가된 데이터프레임
        """
        try:
            self.logger.info("매매 신호 생성 시작")
            
            # 롱 진입 신호
            data['long_signal'] = (
                (data['close'] < data['bb_lower']) &  # 가격이 하단 밴드 아래
                (data['rsi'] < self.rsi_oversold) &    # RSI 과매도
                (data['volume_ratio'] > self.volume_threshold) &  # 거래량 증가
                (data['trend'] == 1)  # 상승 추세
            )
            
            # 숏 진입 신호
            data['short_signal'] = (
                (data['close'] > data['bb_upper']) &  # 가격이 상단 밴드 위
                (data['rsi'] > self.rsi_overbought) &  # RSI 과매수
                (data['volume_ratio'] > self.volume_threshold) &  # 거래량 증가
                (data['trend'] == -1)  # 하락 추세
            )
            
            # 청산 신호
            data['exit_signal'] = (
                (data['close'] > data['bb_middle']) |  # 가격이 중간 밴드 위
                (data['close'] < data['bb_middle']) |  # 가격이 중간 밴드 아래
                (data['rsi'] > 50) |  # RSI 중간값 위
                (data['rsi'] < 50)    # RSI 중간값 아래
            )
            
            self.logger.info("매매 신호 생성 완료")
            return data
            
        except Exception as e:
            self.logger.error(f"매매 신호 생성 중 오류 발생: {str(e)}")
            raise
    
    def should_long(self, data: pd.DataFrame, index: int) -> bool:
        """
        롱 포지션 진입 조건 확인
        
        Args:
            data: OHLCV 데이터프레임
            index: 현재 인덱스
            
        Returns:
            bool: 롱 진입 여부
        """
        return data['long_signal'].iloc[index]
    
    def should_short(self, data: pd.DataFrame, index: int) -> bool:
        """
        숏 포지션 진입 조건 확인
        
        Args:
            data: OHLCV 데이터프레임
            index: 현재 인덱스
            
        Returns:
            bool: 숏 진입 여부
        """
        return data['short_signal'].iloc[index]
    
    def should_exit(self, data: pd.DataFrame, index: int) -> bool:
        """
        포지션 청산 조건 확인
        
        Args:
            data: OHLCV 데이터프레임
            index: 현재 인덱스
            
        Returns:
            bool: 청산 여부
        """
        return data['exit_signal'].iloc[index]
    
    def optimize_parameters(self,
                          data: pd.DataFrame,
                          param_ranges: Dict[str, List[Any]] = None) -> Dict[str, Any]:
        """
        전략 파라미터 최적화
        
        Args:
            data: OHLCV 데이터프레임
            param_ranges: 파라미터 범위 (기본값: None)
            
        Returns:
            Dict: 최적화된 파라미터
        """
        try:
            self.logger.info("파라미터 최적화 시작")
            
            if param_ranges is None:
                param_ranges = {
                    'bb_period': range(10, 30, 5),
                    'bb_std': [1.5, 2.0, 2.5],
                    'rsi_period': range(10, 20, 2),
                    'rsi_oversold': range(20, 35, 5),
                    'rsi_overbought': range(65, 80, 5),
                    'volume_threshold': [1.2, 1.5, 2.0],
                    'trend_period': range(10, 30, 5)
                }
            
            best_params = {}
            best_sharpe = -np.inf
            
            # 파라미터 그리드 서치
            for bb_period in param_ranges['bb_period']:
                for bb_std in param_ranges['bb_std']:
                    for rsi_period in param_ranges['rsi_period']:
                        for rsi_oversold in param_ranges['rsi_oversold']:
                            for rsi_overbought in param_ranges['rsi_overbought']:
                                for volume_threshold in param_ranges['volume_threshold']:
                                    for trend_period in param_ranges['trend_period']:
                                        # 파라미터 설정
                                        self.bb_period = bb_period
                                        self.bb_std = bb_std
                                        self.rsi_period = rsi_period
                                        self.rsi_oversold = rsi_oversold
                                        self.rsi_overbought = rsi_overbought
                                        self.volume_threshold = volume_threshold
                                        self.trend_period = trend_period
                                        
                                        # 백테스트 실행
                                        data_with_indicators = self.calculate_indicators(data.copy())
                                        data_with_signals = self.generate_signals(data_with_indicators)
                                        
                                        # 성과 측정
                                        returns = []
                                        for i in range(len(data_with_signals)):
                                            if self.should_long(data_with_signals, i):
                                                returns.append(data_with_signals['returns'].iloc[i])
                                            elif self.should_short(data_with_signals, i):
                                                returns.append(-data_with_signals['returns'].iloc[i])
                                        
                                        if returns:
                                            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                                            if sharpe_ratio > best_sharpe:
                                                best_sharpe = sharpe_ratio
                                                best_params = {
                                                    'bb_period': bb_period,
                                                    'bb_std': bb_std,
                                                    'rsi_period': rsi_period,
                                                    'rsi_oversold': rsi_oversold,
                                                    'rsi_overbought': rsi_overbought,
                                                    'volume_threshold': volume_threshold,
                                                    'trend_period': trend_period
                                                }
            
            self.logger.info(f"최적화 완료 - 최고 샤프 비율: {best_sharpe:.2f}")
            self.logger.info(f"최적 파라미터: {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"파라미터 최적화 중 오류 발생: {str(e)}")
            raise 