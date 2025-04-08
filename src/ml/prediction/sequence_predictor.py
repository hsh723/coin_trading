import numpy as np
import pandas as pd
from typing import Dict, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class SequencePredictor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'sequence_length': 60,
            'n_features': 10,
            'n_units': 50,
            'dropout_rate': 0.2
        }
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(self.config['n_units'], 
                 return_sequences=True,
                 input_shape=(self.config['sequence_length'], 
                            self.config['n_features'])),
            Dropout(self.config['dropout_rate']),
            LSTM(self.config['n_units'] // 2),
            Dropout(self.config['dropout_rate']),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
