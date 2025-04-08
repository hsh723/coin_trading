import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class MLAnalyzer:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            LSTM(units=50, return_sequences=False),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """데이터 전처리"""
        scaled_data = self.scaler.fit_transform(data[['close']])
        x, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            x.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(x), np.array(y)
