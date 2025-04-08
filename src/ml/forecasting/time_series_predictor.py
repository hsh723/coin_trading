import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TimeSeriesPredictor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
