from typing import List, Dict
from dataclasses import dataclass

@dataclass
class BatchProcessingResult:
    batch_id: str
    processed_count: int
    failed_count: int
    processing_time: float
    errors: List[Dict]

class BatchTaskProcessor:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        
    async def process_batch(self, tasks: List[Dict]) -> BatchProcessingResult:
        """배치 작업 처리"""
        start_time = time.time()
        results = []
        errors = []
        
        for task_group in self._create_batches(tasks):
            try:
                batch_result = await self._process_task_group(task_group)
                results.extend(batch_result)
            except Exception as e:
                errors.append({
                    'error': str(e),
                    'tasks': task_group
                })
                
        return BatchProcessingResult(
            batch_id=self._generate_batch_id(),
            processed_count=len(results),
            failed_count=len(errors),
            processing_time=time.time() - start_time,
            errors=errors
        )
