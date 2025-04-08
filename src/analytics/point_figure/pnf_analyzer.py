from typing import Dict, List
import pandas as pd
import numpy as np

class PointAndFigureAnalyzer:
    def __init__(self, box_size: float = 0.02, reversal_boxes: int = 3):
        self.box_size = box_size
        self.reversal_boxes = reversal_boxes
        
    def analyze_pnf(self, price_data: pd.Series) -> Dict:
        """포인트 앤 피규어 분석"""
        boxes = self._calculate_boxes(price_data)
        signals = self._identify_signals(boxes)
        
        return {
            'current_pattern': self._identify_current_pattern(boxes),
            'support_levels': self._find_support_levels(boxes),
            'resistance_levels': self._find_resistance_levels(boxes),
            'signals': signals
        }
