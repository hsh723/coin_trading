import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    모델 학습 및 예측 시스템
    
    주요 기능:
    - 데이터 준비 및 전처리
    - LSTM 모델 학습
    - 모델 평가 및 검증
    - 예측 수행
    """
    
    def __init__(self,
                 data_dir: str = "./processed_data",
                 model_dir: str = "./models",
                 sequence_length: int = 60,
                 prediction_length: int = 10):
        """
        모델 트레이너 초기화
        
        Args:
            data_dir: 처리된 데이터 디렉토리
            model_dir: 모델 저장 디렉토리
            sequence_length: 시퀀스 길이
            prediction_length: 예측 길이
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 스케일러
        self.scaler = StandardScaler()
        
        # 메트릭스
        self.metrics = {
            'models_trained': 0,
            'training_time': 0,
            'errors': 0
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 준비
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            X: 입력 시퀀스
            y: 타겟 시퀀스
        """
        try:
            # 특성 선택
            features = ['price', 'MA5', 'MA20', 'RSI', 'MACD', 'volatility']
            data = df[features].values
            
            # 데이터 스케일링
            scaled_data = self.scaler.fit_transform(data)
            
            # 시퀀스 생성
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length - self.prediction_length + 1):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length:i + self.sequence_length + self.prediction_length, 0])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"데이터 준비 중 오류 발생: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        LSTM 모델 구축
        
        Args:
            input_shape: 입력 데이터 형태
            
        Returns:
            LSTM 모델
        """
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(self.prediction_length)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            return model
            
        except Exception as e:
            logger.error(f"모델 구축 중 오류 발생: {e}")
            return None
    
    def train_model(self, symbol: str) -> Optional[Sequential]:
        """
        모델 학습
        
        Args:
            symbol: 심볼
            
        Returns:
            학습된 모델
        """
        try:
            start_time = datetime.now()
            
            # 데이터 로드
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith(symbol)]
            if not data_files:
                logger.warning(f"데이터 파일을 찾을 수 없음: {symbol}")
                return None
            
            df = pd.read_csv(os.path.join(self.data_dir, data_files[-1]))
            
            # 데이터 준비
            X, y = self.prepare_data(df)
            if len(X) == 0 or len(y) == 0:
                return None
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 모델 구축
            model = self.build_model((X.shape[1], X.shape[2]))
            if model is None:
                return None
            
            # 콜백 설정
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10),
                ModelCheckpoint(
                    os.path.join(self.model_dir, f"{symbol}_model.h5"),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # 모델 학습
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # 모델 평가
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"모델 평가 - MSE: {mse:.4f}, R2: {r2:.4f}")
            
            # 메트릭스 업데이트
            self.metrics['models_trained'] += 1
            self.metrics['training_time'] += (datetime.now() - start_time).total_seconds()
            
            return model
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {e}")
            self.metrics['errors'] += 1
            return None
    
    def predict(self, model: Sequential, df: pd.DataFrame) -> np.ndarray:
        """
        예측 수행
        
        Args:
            model: 학습된 모델
            df: 입력 데이터프레임
            
        Returns:
            예측 결과
        """
        try:
            # 데이터 준비
            X, _ = self.prepare_data(df)
            if len(X) == 0:
                return np.array([])
            
            # 예측
            predictions = model.predict(X)
            
            # 역스케일링
            predictions = self.scaler.inverse_transform(
                np.hstack([predictions, np.zeros((len(predictions), 5))])
            )[:, 0]
            
            return predictions
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            return np.array([])
    
    def get_metrics(self) -> Dict[str, Any]:
        """학습 메트릭스 반환"""
        return self.metrics 