from typing import Dict, List
from dataclasses import dataclass

@dataclass
class HarmonicPattern:
    pattern_type: str
    completion_level: float
    risk_ratio: float
    entry_price: float
    targets: List[float]

class HarmonicPatternAnalyzer:
    def __init__(self):
        self.patterns = {
            'gartley': {'XA': 0.618, 'AB': 0.382, 'BC': 0.886, 'CD': 1.272},
            'butterfly': {'XA': 0.786, 'AB': 0.382, 'BC': 0.886, 'CD': 1.618},
            'bat': {'XA': 0.886, 'AB': 0.382, 'BC': 0.886, 'CD': 2.0},
            'crab': {'XA': 0.618, 'AB': 0.382, 'BC': 0.886, 'CD': 3.618}
        }
