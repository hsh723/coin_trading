from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class PricePattern:
    pattern_type: str
    reliability: float
    target_price: float
    stop_loss: float
    formation_length: int

class PricePatternAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_pattern_bars': 5,
            'reliability_threshold': 0.7,
            'max_patterns': 3
        }
        
    async def analyze_patterns(self, price_data: np.ndarray) -> List[PricePattern]:
        """가격 패턴 분석"""
        patterns = []
        
        # 헤드앤숄더 패턴 검사
        if head_shoulders := self._detect_head_and_shoulders(price_data):
            patterns.append(head_shoulders)
            
        # 더블 탑/바텀 패턴 검사
        if double_pattern := self._detect_double_pattern(price_data):
            patterns.append(double_pattern)
            
        # 삼각형 패턴 검사
        if triangle := self._detect_triangle_pattern(price_data):
            patterns.append(triangle)
            
        return sorted(patterns, key=lambda x: x.reliability, reverse=True)
