from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class CircuitState:
    is_open: bool
    failure_count: int
    last_failure_time: float
    reset_timeout: float

class TaskCircuitBreaker:
    def __init__(self, threshold: int = 5, reset_timeout: int = 60):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.circuits = {}
        
    async def check_circuit(self, task_id: str) -> bool:
        """서킷 상태 확인"""
        if task_id not in self.circuits:
            self.circuits[task_id] = CircuitState(
                is_open=False,
                failure_count=0,
                last_failure_time=0,
                reset_timeout=self.reset_timeout
            )
            
        circuit = self.circuits[task_id]
        
        if circuit.is_open:
            if time.time() - circuit.last_failure_time > circuit.reset_timeout:
                circuit.is_open = False
                circuit.failure_count = 0
                return True
            return False
            
        return True
