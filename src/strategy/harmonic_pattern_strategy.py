from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class HarmonicPattern:
    pattern_type: str  # Gartley, Butterfly, Bat, Crab
    points: Dict[str, float]
    ratios: Dict[str, float]
    completion: float

class HarmonicPatternStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'pattern_types': ['gartley', 'butterfly', 'bat', 'crab'],
            'tolerance': 0.02,
            'min_pattern_size': 10
        }
        
    async def identify_patterns(self, price_data: np.ndarray) -> List[HarmonicPattern]:
        """하모닉 패턴 식별"""
        patterns = []
        for pattern_type in self.config['pattern_types']:
            if found_pattern := self._find_pattern(price_data, pattern_type):
                patterns.append(found_pattern)
                
        return sorted(patterns, key=lambda x: x.completion, reverse=True)
