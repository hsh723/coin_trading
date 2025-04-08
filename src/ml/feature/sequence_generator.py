from typing import List, Tuple
import numpy as np
import pandas as pd

class SequenceGenerator:
    def __init__(self, sequence_length: int, prediction_length: int = 1):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - self.prediction_length + 1):
            seq = data.iloc[i:i+self.sequence_length]
            target = data.iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_length]
            
            sequences.append(seq.values)
            targets.append(target.values)
            
        return np.array(sequences), np.array(targets)
