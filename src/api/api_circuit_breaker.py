from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class CircuitState:
    is_open: bool
    failure_count: int
    last_failure: float
    reset_timeout: float
    threshold: int

class ApiCircuitBreaker:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'failure_threshold': 5,
            'reset_timeout': 60,
            'half_open_timeout': 30
        }
        self.circuits = {}
        
    async def check_circuit(self, endpoint: str) -> bool:
        """API 서킷 상태 확인"""
        if endpoint not in self.circuits:
            self.circuits[endpoint] = CircuitState(
                is_open=False,
                failure_count=0,
                last_failure=0,
                reset_timeout=self.config['reset_timeout'],
                threshold=self.config['failure_threshold']
            )
            
        return await self._evaluate_circuit_state(endpoint)
