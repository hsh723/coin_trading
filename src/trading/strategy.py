"""
통합 거래 전략 모듈
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.analysis.news_analyzer import news_analyzer
from src.learning.continuous_learner import continuous_learner

logger = logging.getLogger(__name__)

class IntegratedStrategy:
    def __init__(self):
        """
        통합 전략 초기화
        """
        self.parameters = {
            'confidence_threshold': 0.6,
            'risk_factor': 0.02,
            'sentiment_weight': 0.3,
            'technical_weight': 0.7
        }
        
    async def initialize(self) -> None:
        """
        전략 초기화
        """
        try:
            # 지속적 학습 모델 로드
            await continuous_learner.load_model()
            logger.info("통합 전략 초기화 완료")
        except Exception as e:
            logger.error(f"전략 초기화 중 오류 발생: {str(e)}")
    
    def update_parameters(self, **kwargs) -> None:
        """
        전략 파라미터 업데이트
        
        Args:
            **kwargs: 업데이트할 파라미터
        """
        self.parameters.update(kwargs)
        logger.info(f"전략 파라미터 업데이트: {self.parameters}")
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        시장 분석
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 기술적 분석
            technical_score = self._analyze_technical_indicators(market_data)
            
            # 뉴스 감성 분석
            sentiment_score = await news_analyzer.analyze_market_sentiment()
            
            # 머신러닝 예측
            features = self._extract_features(market_data)
            ml_score = continuous_learner.predict_trade(features)
            
            # 종합 점수 계산
            total_score = (
                technical_score * self.parameters['technical_weight'] +
                sentiment_score * self.parameters['sentiment_weight'] +
                ml_score * (1 - self.parameters['technical_weight'] - self.parameters['sentiment_weight'])
            )
            
            analysis_result = {
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'ml_score': ml_score,
                'total_score': total_score,
                'timestamp': datetime.now()
            }
            
            logger.info(f"시장 분석 완료: {analysis_result}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"시장 분석 중 오류 발생: {str(e)}")
            return {
                'technical_score': 0.5,
                'sentiment_score': 0.0,
                'ml_score': 0.5,
                'total_score': 0.5,
                'timestamp': datetime.now()
            }
    
    def _analyze_technical_indicators(self, market_data: Dict[str, Any]) -> float:
        """
        기술적 지표 분석
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            float: 기술적 분석 점수 (0.0 ~ 1.0)
        """
        try:
            # 이동평균선 분석
            ma_score = self._analyze_moving_averages(market_data)
            
            # RSI 분석
            rsi_score = self._analyze_rsi(market_data)
            
            # 볼린저 밴드 분석
            bb_score = self._analyze_bollinger_bands(market_data)
            
            # 종합 점수 계산
            technical_score = (ma_score + rsi_score + bb_score) / 3
            return technical_score
            
        except Exception as e:
            logger.error(f"기술적 지표 분석 중 오류 발생: {str(e)}")
            return 0.5
    
    def _analyze_moving_averages(self, market_data: Dict[str, Any]) -> float:
        """
        이동평균선 분석
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            float: 이동평균선 분석 점수 (0.0 ~ 1.0)
        """
        try:
            # 단기/중기/장기 이동평균선 계산
            short_ma = market_data.get('short_ma', 0)
            mid_ma = market_data.get('mid_ma', 0)
            long_ma = market_data.get('long_ma', 0)
            
            # 골든/데드 크로스 확인
            if short_ma > mid_ma > long_ma:
                return 1.0
            elif short_ma < mid_ma < long_ma:
                return 0.0
            else:
                return 0.5
            
        except Exception as e:
            logger.error(f"이동평균선 분석 중 오류 발생: {str(e)}")
            return 0.5
    
    def _analyze_rsi(self, market_data: Dict[str, Any]) -> float:
        """
        RSI 분석
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            float: RSI 분석 점수 (0.0 ~ 1.0)
        """
        try:
            rsi = market_data.get('rsi', 50)
            
            # RSI 기반 점수 계산
            if rsi > 70:
                return 0.0  # 과매수
            elif rsi < 30:
                return 1.0  # 과매도
            else:
                return 0.5  # 중립
            
        except Exception as e:
            logger.error(f"RSI 분석 중 오류 발생: {str(e)}")
            return 0.5
    
    def _analyze_bollinger_bands(self, market_data: Dict[str, Any]) -> float:
        """
        볼린저 밴드 분석
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            float: 볼린저 밴드 분석 점수 (0.0 ~ 1.0)
        """
        try:
            price = market_data.get('price', 0)
            upper_band = market_data.get('upper_band', 0)
            lower_band = market_data.get('lower_band', 0)
            
            # 볼린저 밴드 기반 점수 계산
            if price > upper_band:
                return 0.0  # 과매수
            elif price < lower_band:
                return 1.0  # 과매도
            else:
                return 0.5  # 중립
            
        except Exception as e:
            logger.error(f"볼린저 밴드 분석 중 오류 발생: {str(e)}")
            return 0.5
    
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        특성 추출
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            np.ndarray: 추출된 특성
        """
        try:
            features = [
                market_data.get('hour', 0),
                market_data.get('day_of_week', 0),
                market_data.get('is_weekend', 0),
                market_data.get('win_rate', 0.5),
                market_data.get('max_drawdown', 0.0)
            ]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"특성 추출 중 오류 발생: {str(e)}")
            return np.zeros(5)
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        매매 신호 생성
        
        Args:
            market_data (Dict[str, Any]): 시장 데이터
            
        Returns:
            Dict[str, Any]: 매매 신호
        """
        try:
            # 시장 분석
            analysis = await self.analyze_market(market_data)
            
            # 매매 신호 생성
            signal = {
                'action': 'hold',
                'confidence': analysis['total_score'],
                'timestamp': datetime.now(),
                'analysis': analysis
            }
            
            # 매수 신호
            if analysis['total_score'] > self.parameters['confidence_threshold']:
                signal['action'] = 'buy'
            
            # 매도 신호
            elif analysis['total_score'] < (1 - self.parameters['confidence_threshold']):
                signal['action'] = 'sell'
            
            logger.info(f"매매 신호 생성: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"매매 신호 생성 중 오류 발생: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.5,
                'timestamp': datetime.now(),
                'analysis': None
            }
    
    async def update(self) -> None:
        """
        전략 업데이트
        """
        try:
            # 지속적 학습 모델 업데이트
            await continuous_learner.update_trading_strategy(self)
            logger.info("전략 업데이트 완료")
        except Exception as e:
            logger.error(f"전략 업데이트 중 오류 발생: {str(e)}")

# 전역 전략 인스턴스
strategy = IntegratedStrategy() 