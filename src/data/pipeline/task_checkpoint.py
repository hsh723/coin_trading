from typing import Dict, Optional
import json
from dataclasses import dataclass

@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    timestamp: float
    state: Dict
    metadata: Dict

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints = {}
        
    async def save_checkpoint(self, task_id: str, state: Dict) -> Checkpoint:
        """작업 체크포인트 저장"""
        checkpoint = Checkpoint(
            checkpoint_id=self._generate_checkpoint_id(),
            task_id=task_id,
            timestamp=time.time(),
            state=state,
            metadata=self._create_metadata()
        )
        
        await self._persist_checkpoint(checkpoint)
        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint
