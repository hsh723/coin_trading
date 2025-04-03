from typing import Dict, Any, Optional
from ..analysis.news_analyzer import NewsAnalyzer
from ..analysis.technical_analyzer import TechnicalAnalyzer
from ..utils.logger import setup_logger
from ..utils.database import DatabaseManager

class IntegratedStrategy:
    """통합 트레이딩 전략 클래스"""
    
    def __init__(self, db: DatabaseManager = None):
        """
        통합 전략 초기화
        
        Args:
            db (DatabaseManager, optional): 데이터베이스 매니저 인스턴스
        """
        self.logger = setup_logger('integrated_strategy')
        self.db = db if db else DatabaseManager()
        self.news_analyzer = NewsAnalyzer(db=self.db)
        self.technical_analyzer = TechnicalAnalyzer(db=self.db)
        
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """거래 신호 생성"""
        try:
            if not market_data or 'ohlcv' not in market_data:
                return {'signal': 'neutral', 'strength': 0.0}
                
            # 기술적 분석 신호
            technical_signal = self.technical_analyzer.analyze(market_data['ohlcv'])
            
            # 뉴스 분석 신호
            news_signal = self.news_analyzer.analyze(market_data['symbol'])
            
            # 신호 통합
            if technical_signal['signal'] == 'buy' and news_signal['sentiment'] > 0:
                return {
                    'signal': 'buy',
                    'strength': (technical_signal['strength'] + news_signal['sentiment']) / 2
                }
            elif technical_signal['signal'] == 'sell' and news_signal['sentiment'] < 0:
                return {
                    'signal': 'sell',
                    'strength': (technical_signal['strength'] + abs(news_signal['sentiment'])) / 2
                }
            else:
                return {'signal': 'neutral', 'strength': 0.0}
                
        except Exception as e:
            self.logger.error(f"거래 신호 생성 실패: {str(e)}")
            return {'signal': 'neutral', 'strength': 0.0}
            
    def calculate_position_size(self, capital: float, risk_per_trade: float, 
                              entry_price: float, stop_loss: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            capital (float): 자본금
            risk_per_trade (float): 거래당 리스크 비율
            entry_price (float): 진입 가격
            stop_loss (float): 손절 가격
            
        Returns:
            float: 포지션 크기
        """
        try:
            # 리스크 금액 계산
            risk_amount = capital * risk_per_trade
            
            # 손절폭 계산
            stop_loss_pips = abs(entry_price - stop_loss)
            
            # 포지션 크기 계산
            position_size = risk_amount / stop_loss_pips
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {str(e)}")
            return 0.0
            
    def calculate_stop_loss(self, entry_price: float, atr: float, 
                          risk_reward_ratio: float = 2.0) -> float:
        """
        손절 가격 계산
        
        Args:
            entry_price (float): 진입 가격
            atr (float): ATR 값
            risk_reward_ratio (float): 리스크:리워드 비율
            
        Returns:
            float: 손절 가격
        """
        try:
            # ATR 기반 손절폭 계산
            stop_loss_distance = atr * 2  # 2 ATR
            
            # 손절 가격 계산
            stop_loss = entry_price - stop_loss_distance
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"손절 가격 계산 실패: {str(e)}")
            return entry_price * 0.95  # 기본 5% 손절
            
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            risk_reward_ratio: float = 2.0) -> float:
        """
        이익 실현 가격 계산
        
        Args:
            entry_price (float): 진입 가격
            stop_loss (float): 손절 가격
            risk_reward_ratio (float): 리스크:리워드 비율
            
        Returns:
            float: 이익 실현 가격
        """
        try:
            # 리스크 크기 계산
            risk = abs(entry_price - stop_loss)
            
            # 리워드 크기 계산
            reward = risk * risk_reward_ratio
            
            # 이익 실현 가격 계산
            take_profit = entry_price + reward
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"이익 실현 가격 계산 실패: {str(e)}")
            return entry_price * 1.1  # 기본 10% 이익 