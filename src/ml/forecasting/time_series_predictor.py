import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import joblib
from typing import Tuple, Dict, Union, List

class TimeSeriesPredictor:
    def __init__(self, sequence_length: int = 60, n_features: int = 1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> Sequential:
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """시계열 데이터를 LSTM 입력 형식으로 준비"""
        # 데이터 스케일링
        values = data[target_col].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        X, y = [], []
        for i in range(len(scaled_values) - self.sequence_length):
            X.append(scaled_values[i:i + self.sequence_length])
            y.append(scaled_values[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_multivariate_data(self, data: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """다변량 시계열 데이터를 LSTM 입력 형식으로 준비"""
        # 타겟 데이터만 스케일링 (예측값을 원래 스케일로 변환하기 위해)
        target_values = data[target_col].values.reshape(-1, 1)
        self.scaler.fit(target_values)
        
        # 모든 피처 스케일링 
        scaled_data = data.copy()
        for column in scaled_data.columns:
            values = scaled_data[column].values.reshape(-1, 1)
            scaled_values = MinMaxScaler().fit_transform(values)
            scaled_data[column] = scaled_values
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data.iloc[i:i + self.sequence_length].values)
            y.append(self.scaler.transform(target_values[i + self.sequence_length].reshape(-1, 1)))
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, 
              patience: int = 10, model_path: str = None) -> None:
        """모델 훈련"""
        callbacks = [EarlyStopping(patience=patience, restore_best_weights=True)]
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        predictions = self.model.predict(X)
        # 스케일 복원
        return self.scaler.inverse_transform(predictions)
    
    def forecast(self, last_sequence: np.ndarray, horizon: int = 1) -> np.ndarray:
        """미래 n 기간 예측"""
        assert last_sequence.shape == (self.sequence_length, self.n_features), \
            f"Input shape {last_sequence.shape} does not match expected shape ({self.sequence_length}, {self.n_features})"
        
        current_sequence = last_sequence.copy()
        forecasts = []
        
        for _ in range(horizon):
            # 단일 시점 예측
            pred = self.model.predict(np.expand_dims(current_sequence, axis=0))
            forecasts.append(pred[0])
            
            # 시퀀스 업데이트 (가장 오래된 값 제거, 새 예측값 추가)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred[0]  # 첫 번째 피처를 예측값으로 설정
        
        forecasts = np.array(forecasts)
        return self.scaler.inverse_transform(forecasts)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """모델 평가 및 성능 지표 반환"""
        y_pred = self.model.predict(X_test)
        
        # 원래 스케일로 변환
        y_test_orig = self.scaler.inverse_transform(y_test)
        y_pred_orig = self.scaler.inverse_transform(y_pred)
        
        # 평가 지표 계산
        metrics = {
            'mse': mean_squared_error(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'r2': r2_score(y_test_orig, y_pred_orig)
        }
        
        return metrics
    
    def plot_history(self) -> None:
        """훈련 히스토리 시각화"""
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = 'Predictions vs Actual') -> None:
        """예측 결과 시각화"""
        plt.figure(figsize=(15, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save(self, model_path: str, scaler_path: str = None) -> None:
        """모델 및 스케일러 저장"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str, sequence_length: int = 60, n_features: int = 1) -> 'TimeSeriesPredictor':
        """저장된 모델 및 스케일러 로드"""
        predictor = cls(sequence_length=sequence_length, n_features=n_features)
        predictor.model = load_model(model_path)
        predictor.scaler = joblib.load(scaler_path)
        return predictor
