from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class DeviationInfo:
    task_id: str
    metric_name: str
    expected_value: float
    actual_value: float
    deviation_ratio: float
    severity: str

class TaskDeviationDetector:
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.baseline_metrics = {}
        
    async def detect_deviations(self, task_id: str, 
                              current_metrics: Dict) -> List[DeviationInfo]:
        """작업 편차 감지"""
        deviations = []
        baseline = self.baseline_metrics.get(task_id, {})
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline:
                expected = baseline[metric_name]
                deviation_ratio = abs(current_value - expected) / expected
                
                if deviation_ratio > self.thresholds.get(metric_name, 0.1):
                    deviations.append(
                        DeviationInfo(
                            task_id=task_id,
                            metric_name=metric_name,
                            expected_value=expected,
                            actual_value=current_value,
                            deviation_ratio=deviation_ratio,
                            severity=self._determine_severity(deviation_ratio)
                        )
                    )
        
        return deviations
