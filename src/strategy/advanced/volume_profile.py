from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class VolumeProfile:
    poc_price: float
    value_areas: Dict[str, float]
    volume_nodes: List[Dict[str, float]]
    distribution_type: str

class VolumeProfileAnalyzer:
    def __init__(self):
        self.config = {
            'value_area_volume': 0.68,
            'num_nodes': 24,
            'min_volume': 1.0
        }
