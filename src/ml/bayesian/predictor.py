import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import json
import os
from datetime import datetime, timedelta

class MarketPredictor:
    """시장 예측"""
    def __init__(self,
                 data_pipeline,
                 config_path: str = "./config/predictor_config.json",
                 model_dir: str = "./models",
                 log_dir: str = "./logs"):
        """
        시장 예측 초기화
        
        Args:
            data_pipeline: 데이터 파이프라인 인스턴스
            config_path: 설정 파일 경로
            model_dir: 모델 저장 디렉토리
            log_dir: 로그 디렉토리
        """
        self.data_pipeline = data_pipeline
        self.config_path = config_path
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # 설정 로드
        self.config = self._load_config()
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        # 모델 및 스케일러
        self.price_model = None
        self.state_model = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
        # 모델 파라미터
        self.seq_length = self.config.get("sequence_length", 60)
        self.batch_size = self.config.get("batch_size", 32)
        self.epochs = self.config.get("epochs", 100)
        self.validation_split = self.config.get("validation_split", 0.2)
        
        # 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
            
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("market_predictor")
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        log_file = os.path.join(self.log_dir, "market_predictor.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
        
    def prepare_data(self,
                    symbol: str,
                    lookback_period: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 준비
        
        Args:
            symbol: 심볼
            lookback_period: 과거 데이터 기간 (일)
            
        Returns:
            특성 데이터, 타겟 데이터
        """
        try:
            # 데이터 수집
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_period)
            df = self.data_pipeline.get_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date
            )
            
            if df.empty:
                raise ValueError("데이터가 없습니다.")
                
            # 기술적 지표 계산
            df = self._calculate_technical_indicators(df)
            
            # 특성 및 타겟 선택
            features = df.drop(['timestamp', 'price'], axis=1).values
            target = df['price'].values
            
            # 데이터 정규화
            self.price_scaler.fit(target.reshape(-1, 1))
            self.feature_scaler.fit(features)
            
            scaled_target = self.price_scaler.transform(target.reshape(-1, 1)).flatten()
            scaled_features = self.feature_scaler.transform(features)
            
            # 시계열 데이터 변환
            X, y = self._create_sequences(scaled_features, scaled_target)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"데이터 준비 중 오류 발생: {e}")
            raise
            
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 계산
        
        Args:
            df: 데이터프레임
            
        Returns:
            지표가 추가된 데이터프레임
        """
        try:
            # 이동평균
            df['MA5'] = df['price'].rolling(window=5).mean()
            df['MA20'] = df['price'].rolling(window=20).mean()
            df['MA60'] = df['price'].rolling(window=60).mean()
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            df['BB_middle'] = df['price'].rolling(window=20).mean()
            df['BB_std'] = df['price'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
            
            # MACD
            exp1 = df['price'].ewm(span=12, adjust=False).mean()
            exp2 = df['price'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 거래량 가중 평균 가격
            df['VWAP'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # 결측치 처리
            df = df.fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 중 오류 발생: {e}")
            raise
            
    def _create_sequences(self,
                        features: np.ndarray,
                        target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터 변환
        
        Args:
            features: 특성 데이터
            target: 타겟 데이터
            
        Returns:
            시퀀스 데이터, 타겟 데이터
        """
        try:
            X, y = [], []
            for i in range(len(features) - self.seq_length):
                X.append(features[i:(i + self.seq_length)])
                y.append(target[i + self.seq_length])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"시퀀스 생성 중 오류 발생: {e}")
            raise
            
    def build_price_model(self) -> Sequential:
        """가격 예측 모델 구축"""
        try:
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.seq_length, len(self.feature_columns))),
                BatchNormalization(),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),
                LSTM(32),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
            
        except Exception as e:
            self.logger.error(f"가격 예측 모델 구축 중 오류 발생: {e}")
            return None
            
    def build_state_model(self) -> Sequential:
        """시장 상태 분류 모델 구축"""
        try:
            model = Sequential([
                GRU(128, return_sequences=True, input_shape=(self.seq_length, len(self.feature_columns))),
                BatchNormalization(),
                Dropout(0.2),
                GRU(64),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # 상승, 하락, 횡보
            ])
            
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
            
        except Exception as e:
            self.logger.error(f"시장 상태 모델 구축 중 오류 발생: {e}")
            return None
            
    def train_price_model(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         save_best: bool = True) -> Dict[str, Any]:
        """
        가격 예측 모델 학습
        
        Args:
            X: 입력 데이터
            y: 타겟 데이터
            save_best: 최적 모델 저장 여부
            
        Returns:
            학습 결과 딕셔너리
        """
        try:
            # 데이터 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, shuffle=False
            )
            
            # 모델 구축
            self.price_model = self.build_price_model()
            if self.price_model is None:
                raise Exception("모델 구축 실패")
                
            # 콜백 설정
            callbacks = []
            if save_best:
                checkpoint = ModelCheckpoint(
                    os.path.join(self.model_dir, "best_price_model.h5"),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(checkpoint)
                
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
            # 모델 학습
            history = self.price_model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # 학습 결과 저장
            results = {
                "train_loss": history.history['loss'],
                "val_loss": history.history['val_loss'],
                "train_mae": history.history['mae'],
                "val_mae": history.history['val_mae'],
                "final_val_loss": history.history['val_loss'][-1],
                "final_val_mae": history.history['val_mae'][-1]
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"가격 예측 모델 학습 중 오류 발생: {e}")
            return {}
            
    def train_state_model(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         save_best: bool = True) -> Dict[str, Any]:
        """
        시장 상태 모델 학습
        
        Args:
            X: 입력 데이터
            y: 타겟 데이터 (원-핫 인코딩된 상태)
            save_best: 최적 모델 저장 여부
            
        Returns:
            학습 결과 딕셔너리
        """
        try:
            # 데이터 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, shuffle=False
            )
            
            # 모델 구축
            self.state_model = self.build_state_model()
            if self.state_model is None:
                raise Exception("모델 구축 실패")
                
            # 콜백 설정
            callbacks = []
            if save_best:
                checkpoint = ModelCheckpoint(
                    os.path.join(self.model_dir, "best_state_model.h5"),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(checkpoint)
                
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
            # 모델 학습
            history = self.state_model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # 학습 결과 저장
            results = {
                "train_loss": history.history['loss'],
                "val_loss": history.history['val_loss'],
                "train_accuracy": history.history['accuracy'],
                "val_accuracy": history.history['val_accuracy'],
                "final_val_loss": history.history['val_loss'][-1],
                "final_val_accuracy": history.history['val_accuracy'][-1]
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"시장 상태 모델 학습 중 오류 발생: {e}")
            return {}
            
    def predict_price(self, data: np.ndarray) -> Tuple[float, float]:
        """
        가격 예측
        
        Args:
            data: 입력 데이터
            
        Returns:
            예측 가격, 신뢰도
        """
        try:
            if self.price_model is None:
                raise Exception("모델이 로드되지 않음")
                
            # 데이터 전처리
            scaled_data = self.feature_scaler.transform(data)
            X = scaled_data[-self.seq_length:].reshape(1, self.seq_length, -1)
            
            # 예측
            scaled_pred = self.price_model.predict(X)[0][0]
            pred_price = self.price_scaler.inverse_transform([[scaled_pred]])[0][0]
            
            # 신뢰도 계산 (표준편차 기반)
            pred_std = np.std(self.price_model.predict(X))
            confidence = 1 - (pred_std / np.mean(self.price_scaler.scale_))
            
            return pred_price, confidence
            
        except Exception as e:
            self.logger.error(f"가격 예측 중 오류 발생: {e}")
            return None, None
            
    def predict_state(self, data: np.ndarray) -> Tuple[str, float]:
        """
        시장 상태 예측
        
        Args:
            data: 입력 데이터
            
        Returns:
            예측 상태, 확률
        """
        try:
            if self.state_model is None:
                raise Exception("모델이 로드되지 않음")
                
            # 데이터 전처리
            scaled_data = self.feature_scaler.transform(data)
            X = scaled_data[-self.seq_length:].reshape(1, self.seq_length, -1)
            
            # 예측
            pred_probs = self.state_model.predict(X)[0]
            state_idx = np.argmax(pred_probs)
            states = ["상승", "하락", "횡보"]
            
            return states[state_idx], pred_probs[state_idx]
            
        except Exception as e:
            self.logger.error(f"시장 상태 예측 중 오류 발생: {e}")
            return None, None
            
    def save_models(self) -> bool:
        """모델 저장"""
        try:
            if self.price_model is not None:
                self.price_model.save(os.path.join(self.model_dir, "price_model.h5"))
            if self.state_model is not None:
                self.state_model.save(os.path.join(self.model_dir, "state_model.h5"))
            return True
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {e}")
            return False
            
    def load_models(self) -> bool:
        """모델 로드"""
        try:
            price_model_path = os.path.join(self.model_dir, "best_price_model.h5")
            state_model_path = os.path.join(self.model_dir, "best_state_model.h5")
            
            if os.path.exists(price_model_path):
                self.price_model = load_model(price_model_path)
            if os.path.exists(state_model_path):
                self.state_model = load_model(state_model_path)
                
            return True
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            return False
            
    def evaluate_models(self,
                       X: np.ndarray,
                       y_price: np.ndarray,
                       y_state: np.ndarray) -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            X: 입력 데이터
            y_price: 가격 타겟
            y_state: 상태 타겟
            
        Returns:
            평가 결과 딕셔너리
        """
        try:
            results = {}
            
            # 가격 모델 평가
            if self.price_model is not None:
                price_metrics = self.price_model.evaluate(X, y_price, verbose=0)
                results["price_model"] = {
                    "loss": price_metrics[0],
                    "mae": price_metrics[1]
                }
                
            # 상태 모델 평가
            if self.state_model is not None:
                state_metrics = self.state_model.evaluate(X, y_state, verbose=0)
                results["state_model"] = {
                    "loss": state_metrics[0],
                    "accuracy": state_metrics[1]
                }
                
            return results
            
        except Exception as e:
            self.logger.error(f"모델 평가 중 오류 발생: {e}")
            return {} 