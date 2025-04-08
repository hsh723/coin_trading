import tensorflow as tf
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    sequence_length: int = 60
    n_features: int = 10
    n_layers: int = 2
    hidden_units: int = 50
    dropout_rate: float = 0.2

class LSTMModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """LSTM 모델 구축"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(self.config.hidden_units, 
                               return_sequences=True,
                               input_shape=(self.config.sequence_length, self.config.n_features)),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
