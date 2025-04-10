import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class MergeResult:
    success: bool
    merged_data: pd.DataFrame
    conflict_count: int
    resolution_method: str

class DataMerger:
    def __init__(self, merge_strategy: str = 'latest'):
        self.merge_strategy = merge_strategy
        
    async def merge_datasets(self, datasets: List[pd.DataFrame]) -> MergeResult:
        """데이터셋 병합"""
        try:
            merged = self._apply_merge_strategy(datasets)
            conflicts = self._detect_conflicts(merged)
            resolved = self._resolve_conflicts(merged, conflicts)
            
            return MergeResult(
                success=True,
                merged_data=resolved,
                conflict_count=len(conflicts),
                resolution_method=self.merge_strategy
            )
        except Exception as e:
            self._handle_merge_error(e)
