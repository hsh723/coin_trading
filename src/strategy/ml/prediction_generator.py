from typing import Dict
import numpy as np

class PredictionGenerator:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
    async def generate_predictions(self, 
                                 model: Any, 
                                 features: np.ndarray) -> Dict:
        """예측 생성"""
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        return {
            'predictions': predictions,
            'confidence': np.max(probabilities, axis=1),
            'direction': np.where(predictions > 0, 'up', 'down'),
            'signal_strength': self._calculate_signal_strength(probabilities)
        }
