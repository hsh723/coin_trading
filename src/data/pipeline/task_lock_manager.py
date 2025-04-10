from typing import Dict, Optional
import asyncio
import time
from dataclasses import dataclass

@dataclass
class LockInfo:
    task_id: str
    lock_id: str
    acquired_time: float
    expires_at: float
    owner: str

class TaskLockManager:
    def __init__(self, lock_timeout: int = 300):
        self.lock_timeout = lock_timeout
        self.locks: Dict[str, LockInfo] = {}
        
    async def acquire_lock(self, task_id: str, owner: str) -> Optional[str]:
        """작업 잠금 획득"""
        if task_id in self.locks:
            if not self._is_lock_expired(task_id):
                return None
            
        lock_id = self._generate_lock_id()
        self.locks[task_id] = LockInfo(
            task_id=task_id,
            lock_id=lock_id,
            acquired_time=time.time(),
            expires_at=time.time() + self.lock_timeout,
            owner=owner
        )
        
        return lock_id
