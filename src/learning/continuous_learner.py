"""
지속적 학습 시스템 모듈
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.database import db_manager

logger = logging.getLogger(__name__)

class ContinuousLearner:
    def __init__(self):
        """
        지속적 학습 시스템 초기화
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.last_training_time = None
        self.min_training_interval = timedelta(hours=1)
        
    async def prepare_training_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        학습 데이터 준비
        
        Args:
            days (int): 조회할 기간 (일)
            
        Returns:
            Optional[pd.DataFrame]: 학습 데이터
        """
        try:
            # 거래 내역 조회
            trades = await db_manager.get_trades(
                start_time=datetime.now() - timedelta(days=days)
            )
            
            if not trades:
                logger.warning("학습할 거래 데이터가 없습니다.")
                return None
            
            # 데이터프레임 생성
            df = pd.DataFrame(trades)
            
            # 특성 생성
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # 성과 데이터 조회
            performance = await db_manager.get_performance(
                start_time=datetime.now() - timedelta(days=days)
            )
            
            if performance:
                perf_df = pd.DataFrame(performance)
                df = df.merge(
                    perf_df[['timestamp', 'win_rate', 'max_drawdown']],
                    on='timestamp',
                    how='left'
                )
            
            # 결측치 처리
            df = df.fillna(method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"학습 데이터 준비 중 오류 발생: {str(e)}")
            return None
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        특성 추출
        
        Args:
            df (pd.DataFrame): 원본 데이터
            
        Returns:
            np.ndarray: 추출된 특성
        """
        features = [
            'hour',
            'day_of_week',
            'is_weekend',
            'win_rate',
            'max_drawdown'
        ]
        
        return df[features].values
    
    def prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        레이블 준비
        
        Args:
            df (pd.DataFrame): 원본 데이터
            
        Returns:
            np.ndarray: 레이블
        """
        # 수익이 난 거래는 1, 손실이 난 거래는 0
        return (df['pnl'] > 0).astype(int)
    
    async def train_model(self) -> bool:
        """
        모델 학습
        
        Returns:
            bool: 학습 성공 여부
        """
        try:
            # 학습 간격 확인
            if (self.last_training_time and 
                datetime.now() - self.last_training_time < self.min_training_interval):
                logger.info("최소 학습 간격이 지나지 않았습니다.")
                return False
            
            # 데이터 준비
            df = await self.prepare_training_data()
            if df is None:
                return False
            
            # 특성과 레이블 추출
            X = self.extract_features(df)
            y = self.prepare_labels(df)
            
            # 특성 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 모델 학습
            self.model.fit(X_scaled, y)
            self.last_training_time = datetime.now()
            
            # 모델 성능 평가
            score = self.model.score(X_scaled, y)
            logger.info(f"모델 학습 완료 (정확도: {score:.2f})")
            
            # 모델 저장
            await self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            return False
    
    async def save_model(self) -> None:
        """
        모델 저장
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'last_training_time': self.last_training_time
            }
            
            await db_manager.save_setting('trading_model', model_data)
            logger.info("모델 저장 완료")
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {str(e)}")
    
    async def load_model(self) -> bool:
        """
        모델 로드
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            model_data = await db_manager.get_setting('trading_model')
            
            if model_data:
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.last_training_time = model_data['last_training_time']
                logger.info("모델 로드 완료")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False
    
    def predict_trade(self, features: np.ndarray) -> float:
        """
        거래 예측
        
        Args:
            features (np.ndarray): 입력 특성
            
        Returns:
            float: 예측 확률 (0.0 ~ 1.0)
        """
        try:
            # 특성 스케일링
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 예측 확률 계산
            prob = self.model.predict_proba(features_scaled)[0][1]
            return float(prob)
            
        except Exception as e:
            logger.error(f"거래 예측 중 오류 발생: {str(e)}")
            return 0.5
    
    async def update_trading_strategy(self, strategy: Any) -> None:
        """
        거래 전략 업데이트
        
        Args:
            strategy: 거래 전략 객체
        """
        try:
            # 모델 학습
            if await self.train_model():
                # 전략 파라미터 업데이트
                strategy.update_parameters(
                    confidence_threshold=0.6,
                    risk_factor=0.02
                )
                logger.info("거래 전략 업데이트 완료")
            
        except Exception as e:
            logger.error(f"거래 전략 업데이트 중 오류 발생: {str(e)}")

# 전역 지속적 학습 시스템 인스턴스
continuous_learner = ContinuousLearner() 