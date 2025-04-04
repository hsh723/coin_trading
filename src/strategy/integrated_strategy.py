"""
통합 트레이딩 전략 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

class IntegratedStrategy:
    """통합 트레이딩 전략 클래스"""
    
    def __init__(self):
        """통합 전략 초기화"""
        self.logger = logging.getLogger(__name__)
        self.indicators = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # RSI 계산
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD 계산
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = exp1 - exp2
            data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            
            # 볼린저 밴드 계산
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
            data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
            
            return data
            
        except Exception as e:
            self.logger.error(f"지표 계산 실패: {str(e)}")
            return data
            
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """거래 신호 생성"""
        try:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0  # 0: 보유, 1: 매수, -1: 매도
            
            # RSI 신호
            signals.loc[data['rsi'] < 30, 'signal'] = 1  # 과매도
            signals.loc[data['rsi'] > 70, 'signal'] = -1  # 과매수
            
            # MACD 신호
            signals.loc[data['macd'] > data['signal'], 'signal'] = 1
            signals.loc[data['macd'] < data['signal'], 'signal'] = -1
            
            # 볼린저 밴드 신호
            signals.loc[data['close'] < data['bb_lower'], 'signal'] = 1
            signals.loc[data['close'] > data['bb_upper'], 'signal'] = -1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"신호 생성 실패: {str(e)}")
            return pd.DataFrame()
            
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """시장 분석"""
        try:
            analysis = {
                'trend': 'neutral',
                'strength': 0.0,
                'volatility': 0.0,
                'momentum': 0.0
            }
            
            # 추세 분석
            if data['close'].iloc[-1] > data['close'].iloc[-20]:
                analysis['trend'] = 'up'
            elif data['close'].iloc[-1] < data['close'].iloc[-20]:
                analysis['trend'] = 'down'
                
            # 변동성 분석
            analysis['volatility'] = data['close'].pct_change().std()
            
            # 모멘텀 분석
            analysis['momentum'] = data['close'].pct_change(periods=5).mean()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"시장 분석 실패: {str(e)}")
            return {}
            
    def get_position_size(self, 
                         capital: float,
                         risk_per_trade: float,
                         stop_loss: float) -> float:
        """포지션 크기 계산"""
        try:
            risk_amount = capital * risk_per_trade
            position_size = risk_amount / abs(stop_loss)
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            return 0.0
            
    def update_parameters(self, params: Dict) -> None:
        """전략 파라미터 업데이트"""
        try:
            self.indicators.update(params)
            self.logger.info(f"전략 파라미터 업데이트: {params}")
            
        except Exception as e:
            self.logger.error(f"파라미터 업데이트 실패: {str(e)}")
            
    def get_current_signals(self) -> Dict:
        """현재 거래 신호 조회"""
        return {
            'timestamp': datetime.now().isoformat(),
            'signals': {
                'rsi': self.indicators.get('rsi', 0),
                'macd': self.indicators.get('macd', 0),
                'bb': self.indicators.get('bb', 0)
            }
        } 