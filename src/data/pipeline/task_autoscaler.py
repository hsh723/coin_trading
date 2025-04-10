from typing import Dict
from dataclasses import dataclass

@dataclass
class ScalingDecision:
    scale_up: bool
    scale_down: bool
    target_workers: int
    reason: str

class TaskAutoscaler:
    def __init__(self, scaling_config: Dict = None):
        self.config = scaling_config or {
            'min_workers': 1,
            'max_workers': 10,
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.2
        }
        
    async def evaluate_scaling(self, metrics: Dict) -> ScalingDecision:
        """스케일링 결정"""
        current_load = metrics.get('worker_load', 0)
        current_workers = metrics.get('current_workers', 1)
        
        if current_load > self.config['scale_up_threshold']:
            return self._scale_up_decision(current_workers)
        elif current_load < self.config['scale_down_threshold']:
            return self._scale_down_decision(current_workers)
            
        return ScalingDecision(
            scale_up=False,
            scale_down=False,
            target_workers=current_workers,
            reason="No scaling needed"
        )
