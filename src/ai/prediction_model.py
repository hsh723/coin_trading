"""
AI 예측 모델 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class PricePredictionModel:
    """가격 예측 모델 클래스"""
    
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
            # 기술적 지표 계산
            df = data.copy()
            
            # 이동평균선
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # 거래량 지표
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 가격 변동성
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # 목표 변수 (다음 기간 수익률)
            df['target'] = df['close'].pct_change().shift(-1)
            
            # 결측치 제거
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"특징 데이터 준비 실패: {str(e)}")
            return pd.DataFrame()
            
    def train_model(self,
                   data: pd.DataFrame,
                   params: Optional[Dict] = None) -> Dict:
        """
        모델 학습
        
        Args:
            data (pd.DataFrame): 학습 데이터
            params (Optional[Dict]): 모델 파라미터
            
        Returns:
            Dict: 학습 결과
        """
        try:
            # 특징 데이터 준비
            df = self.prepare_features(data)
            
            if df.empty:
                raise ValueError("학습 데이터가 비어있습니다.")
                
            # 특징과 목표 변수 분리
            X = df.drop(['target'], axis=1)
            y = df['target']
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # 데이터 스케일링
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 기본 파라미터 설정
            if params is None:
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
            
            # LightGBM 데이터셋 생성
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            
            # 모델 학습
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, test_data],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # 예측 및 평가
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 모델 저장
            self._save_model()
            
            return {
                'mse': mse,
                'r2': r2,
                'feature_importance': dict(zip(X.columns, self.model.feature_importance()))
            }
            
        except Exception as e:
            self.logger.error(f"모델 학습 실패: {str(e)}")
            return {}
            
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        가격 변동 예측
        
        Args:
            data (pd.DataFrame): 예측 데이터
            
        Returns:
            Dict: 예측 결과
        """
        try:
            if self.model is None:
                self._load_model()
                
            if self.model is None:
                raise ValueError("학습된 모델이 없습니다.")
                
            # 특징 데이터 준비
            df = self.prepare_features(data)
            
            if df.empty:
                raise ValueError("예측 데이터가 비어있습니다.")
                
            # 특징과 목표 변수 분리
            X = df.drop(['target'], axis=1)
            
            # 데이터 스케일링
            X_scaled = self.scaler.transform(X)
            
            # 예측
            predictions = self.model.predict(X_scaled)
            
            # 예측 결과 분석
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            confidence = 1 - (std_pred / abs(mean_pred)) if mean_pred != 0 else 0
            
            return {
                'predictions': predictions,
                'mean_prediction': mean_pred,
                'std_prediction': std_pred,
                'confidence': confidence,
                'direction': 'up' if mean_pred > 0 else 'down',
                'strength': abs(mean_pred)
            }
            
        except Exception as e:
            self.logger.error(f"가격 변동 예측 실패: {str(e)}")
            return {}
            
    def _save_model(self):
        """모델 저장"""
        try:
            if self.model is not None:
                model_path = os.path.join(self.model_dir, 'price_prediction_model.txt')
                self.model.save_model(model_path)
                
                scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
                
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {str(e)}")
            
    def _load_model(self):
        """모델 로드"""
        try:
            model_path = os.path.join(self.model_dir, 'price_prediction_model.txt')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = lgb.Booster(model_file=model_path)
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