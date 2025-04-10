from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class DeduplicationResult:
    original_count: int
    duplicate_count: int
    unique_count: int
    removed_records: List[Dict]

class DataDeduplicator:
    def __init__(self, dedup_config: Dict = None):
        self.config = dedup_config or {
            'columns': ['timestamp', 'symbol', 'price'],
            'tolerance': 0.0001
        }
        
    async def remove_duplicates(self, data: pd.DataFrame) -> DeduplicationResult:
        """데이터 중복 제거"""
        original_count = len(data)
        unique_data = data.drop_duplicates(
            subset=self.config['columns'],
            keep='first'
        )
        
        return DeduplicationResult(
            original_count=original_count,
            duplicate_count=original_count - len(unique_data),
            unique_count=len(unique_data),
            removed_records=self._get_removed_records(data, unique_data)
        )
