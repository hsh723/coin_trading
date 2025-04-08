"""
시장 분류기 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MarketClassifier:
    """시장 분류기 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.model_dir = 'models'
        self._init_model_dir()
        
    def _init_model_dir(self):
        """모델 디렉토리 초기화"""
        try:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        except Exception as e:
            self.logger.error(f"모델 디렉토리 초기화 실패: {str(e)}")
            
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        특징 데이터 준비
        
        Args:
            data (pd.DataFrame): OHLCV 데이터
            
        Returns:
            pd.DataFrame: 특징 데이터
        """
        try:
            df = data.copy()
            
            # 가격 변동성
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # 추세 지표
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            df['ma200'] = df['close'].rolling(window=200).mean()
            
            # 추세 강도
            df['trend_strength'] = abs(df['ma20'] - df['ma200']) / df['ma200']
            
            # 거래량 지표
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 가격 범위
            df['range'] = (df['high'] - df['low']) / df['close']
            df['range_ma'] = df['range'].rolling(window=20).mean()
            
            # 모멘텀
            df['momentum'] = df['close'] / df['close'].shift(20) - 1
            
            # 결측치 제거
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"특징 데이터 준비 실패: {str(e)}")
            return pd.DataFrame()
            
    def label_market_conditions(self, data: pd.DataFrame) -> pd.Series:
        """
        시장 상태 레이블링
        
        Args:
            data (pd.DataFrame): 특징 데이터
            
        Returns:
            pd.Series: 시장 상태 레이블
        """
        try:
            labels = []
            
            for i in range(len(data)):
                # 추세 판단
                trend = 'uptrend' if data.iloc[i]['ma20'] > data.iloc[i]['ma200'] else 'downtrend'
                
                # 변동성 판단
                volatility = 'high' if data.iloc[i]['volatility'] > data['volatility'].mean() else 'low'
                
                # 거래량 판단
                volume = 'high' if data.iloc[i]['volume_ratio'] > 1.5 else 'low'
                
                # 시장 상태 분류
                if trend == 'uptrend' and volatility == 'low':
                    label = 'strong_uptrend'
                elif trend == 'uptrend' and volatility == 'high':
                    label = 'volatile_uptrend'
                elif trend == 'downtrend' and volatility == 'low':
                    label = 'strong_downtrend'
                elif trend == 'downtrend' and volatility == 'high':
                    label = 'volatile_downtrend'
                elif abs(data.iloc[i]['trend_strength']) < 0.02:
                    label = 'sideways'
                else:
                    label = 'transition'
                    
                labels.append(label)
                
            return pd.Series(labels, index=data.index)
            
        except Exception as e:
            self.logger.error(f"시장 상태 레이블링 실패: {str(e)}")
            return pd.Series()
            
    def train_model(self, data: pd.DataFrame) -> Dict:
        """
        모델 학습
        
        Args:
            data (pd.DataFrame): 학습 데이터
            
        Returns:
            Dict: 학습 결과
        """
        try:
            # 특징 데이터 준비
            df = self.prepare_features(data)
            
            if df.empty:
                raise ValueError("학습 데이터가 비어있습니다.")
                
            # 시장 상태 레이블링
            labels = self.label_market_conditions(df)
            
            # 특징과 레이블 분리
            X = df.drop(['returns'], axis=1)
            y = labels
            
            # 데이터 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 모델 학습
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # 모델 저장
            self._save_model()
            
            return {
                'accuracy': self.model.score(X_scaled, y),
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
            }
            
        except Exception as e:
            self.logger.error(f"모델 학습 실패: {str(e)}")
            return {}
            
    def classify_market(self, data: pd.DataFrame) -> Dict:
        """
        시장 상태 분류
        
        Args:
            data (pd.DataFrame): 분류 데이터
            
        Returns:
            Dict: 분류 결과
        """
        try:
            if self.model is None:
                self._load_model()
                
            if self.model is None:
                raise ValueError("학습된 모델이 없습니다.")
                
            # 특징 데이터 준비
            df = self.prepare_features(data)
            
            if df.empty:
                raise ValueError("분류 데이터가 비어있습니다.")
                
            # 특징과 레이블 분리
            X = df.drop(['returns'], axis=1)
            
            # 데이터 스케일링
            X_scaled = self.scaler.transform(X)
            
            # 시장 상태 분류
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # 최적 전략 선택
            recommended_strategies = self._select_optimal_strategies(
                predictions[-1],
                probabilities[-1]
            )
            
            return {
                'current_market_state': predictions[-1],
                'state_probabilities': dict(zip(self.model.classes_, probabilities[-1])),
                'recommended_strategies': recommended_strategies,
                'market_features': {
                    'trend_strength': float(df.iloc[-1]['trend_strength']),
                    'volatility': float(df.iloc[-1]['volatility']),
                    'volume_ratio': float(df.iloc[-1]['volume_ratio'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"시장 상태 분류 실패: {str(e)}")
            return {}
            
    def _select_optimal_strategies(self,
                                 market_state: str,
                                 probabilities: np.ndarray) -> List[str]:
        """
        최적 전략 선택
        
        Args:
            market_state (str): 현재 시장 상태
            probabilities (np.ndarray): 상태 확률
            
        Returns:
            List[str]: 추천 전략 목록
        """
        try:
            # 시장 상태별 전략 매핑
            strategy_mapping = {
                'strong_uptrend': ['momentum', 'breakout'],
                'volatile_uptrend': ['swing', 'mean_reversion'],
                'strong_downtrend': ['short', 'breakdown'],
                'volatile_downtrend': ['swing', 'mean_reversion'],
                'sideways': ['range', 'mean_reversion'],
                'transition': ['wait', 'monitor']
            }
            
            # 기본 전략 선택
            strategies = strategy_mapping.get(market_state, ['wait'])
            
            # 확률 기반 전략 조정
            if market_state in ['volatile_uptrend', 'volatile_downtrend']:
                if probabilities.max() < 0.6:
                    strategies.append('reduce_position')
                    
            elif market_state == 'transition':
                if probabilities.max() < 0.5:
                    strategies = ['wait', 'monitor']
                    
            return strategies
            
        except Exception as e:
            self.logger.error(f"최적 전략 선택 실패: {str(e)}")
            return ['wait']
            
    def _save_model(self):
        """모델 저장"""
        try:
            if self.model is not None:
                model_path = os.path.join(self.model_dir, 'market_classifier.pkl')
                scaler_path = os.path.join(self.model_dir, 'market_scaler.pkl')
                
                joblib.dump(self.model, model_path)
                joblib.dump(self.scaler, scaler_path)
                
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {str(e)}")
            
    def _load_model(self):
        """모델 로드"""
        try:
            model_path = os.path.join(self.model_dir, 'market_classifier.pkl')
            scaler_path = os.path.join(self.model_dir, 'market_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            
    def update_model(self,
                    new_data: pd.DataFrame,
                    update_interval: int = 7) -> bool:
        """
        모델 업데이트
        
        Args:
            new_data (pd.DataFrame): 새로운 데이터
            update_interval (int): 업데이트 간격(일)
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 마지막 업데이트 시간 확인
            last_update_path = os.path.join(self.model_dir, 'last_update.txt')
            
            if os.path.exists(last_update_path):
                with open(last_update_path, 'r') as f:
                    last_update = datetime.fromisoformat(f.read())
                    
                if (datetime.now() - last_update).days < update_interval:
                    return False
                    
            # 모델 업데이트
            self.train_model(new_data)
            
            # 업데이트 시간 저장
            with open(last_update_path, 'w') as f:
                f.write(datetime.now().isoformat())
                
            return True
            
        except Exception as e:
            self.logger.error(f"모델 업데이트 실패: {str(e)}")
            return False 