from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class RecoveryState:
    task_id: str
    recovery_attempt: int
    last_known_state: Dict
    recovery_strategy: str

class TaskRecoveryManager:
    def __init__(self, recovery_config: Dict = None):
        self.config = recovery_config or {
            'max_attempts': 3,
            'strategies': ['retry', 'checkpoint', 'rollback']
        }
        self.recovery_history = {}
        
    async def recover_task(self, task_id: str, error: Exception) -> Optional[Dict]:
        """작업 복구 시도"""
        if task_id not in self.recovery_history:
            self.recovery_history[task_id] = []
            
        recovery_state = RecoveryState(
            task_id=task_id,
            recovery_attempt=len(self.recovery_history[task_id]) + 1,
            last_known_state=await self._get_last_known_state(task_id),
            recovery_strategy=self._determine_recovery_strategy(error)
        )
        
        try:
            if await self._execute_recovery(recovery_state):
                return {'success': True, 'state': recovery_state}
        except Exception as e:
            self.recovery_history[task_id].append({
                'timestamp': time.time(),
                'error': str(e),
                'state': recovery_state
            })
            
        return None
