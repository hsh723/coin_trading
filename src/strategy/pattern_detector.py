from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class PatternDetection:
    pattern_name: str
    confidence: float
    entry_points: List[float]
    target_prices: List[float]
    stop_loss: float

class PatternDetector:
    def __init__(self, pattern_config: Dict = None):
        self.config = pattern_config or {
            'min_pattern_size': 5,
            'confidence_threshold': 0.7,
            'pattern_types': ['double_top', 'double_bottom', 'head_shoulders']
        }
        
    async def detect_patterns(self, price_data: np.ndarray) -> List[PatternDetection]:
        """차트 패턴 감지"""
        patterns = []
        
        for pattern_type in self.config['pattern_types']:
            if pattern := self._detect_specific_pattern(price_data, pattern_type):
                patterns.append(pattern)
                
        return sorted(patterns, key=lambda x: x.confidence, reverse=True)
