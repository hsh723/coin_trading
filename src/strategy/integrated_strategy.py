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
        
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        거래 신호 생성
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            Optional[Dict[str, Any]]: 거래 신호
        """
        try:
            # 기술적 분석
            df = market_data.get('ohlcv')
            if df is None or df.empty:
                self.logger.warning("OHLCV 데이터가 없습니다.")
                return None
                
            # 기술적 지표 계산
            df = self.technical_analyzer.calculate_indicators(df)
            
            # 기술적 분석 신호
            technical_signal = self.technical_analyzer.generate_signals(df)
            
            # 뉴스 분석
            news_sentiment = self.news_analyzer.get_market_sentiment('BTC')
            
            # 신호 통합
            final_signal = self._integrate_signals(technical_signal, news_sentiment)
            
            if final_signal['signal'] != 'neutral':
                self.logger.info(f"거래 신호 생성: {final_signal}")
                
            return final_signal
            
        except Exception as e:
            self.logger.error(f"거래 신호 생성 실패: {str(e)}")
            return None
            
    def _integrate_signals(self, technical: Dict[str, Any], news: Dict[str, Any]) -> Dict[str, Any]:
        """
        기술적 분석과 뉴스 분석 신호 통합
        
        Args:
            technical (Dict[str, Any]): 기술적 분석 신호
            news (Dict[str, Any]): 뉴스 분석 신호
            
        Returns:
            Dict[str, Any]: 통합된 거래 신호
        """
        try:
            # 기술적 분석 가중치
            tech_weight = 0.7
            # 뉴스 분석 가중치
            news_weight = 0.3
            
            # 기술적 분석 점수
            tech_score = 0.0
            if technical['signal'] == 'buy':
                tech_score = technical['strength']
            elif technical['signal'] == 'sell':
                tech_score = -technical['strength']
                
            # 뉴스 분석 점수
            news_score = news['sentiment_score']
            
            # 최종 점수 계산
            final_score = (tech_score * tech_weight) + (news_score * news_weight)
            
            # 신호 결정
            if final_score > 0.3:
                signal = 'buy'
                strength = min(final_score, 1.0)
            elif final_score < -0.3:
                signal = 'sell'
                strength = min(abs(final_score), 1.0)
            else:
                signal = 'neutral'
                strength = 0.0
                
            return {
                'signal': signal,
                'strength': strength,
                'technical': technical,
                'news': news
            }
            
        except Exception as e:
            self.logger.error(f"신호 통합 실패: {str(e)}")
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