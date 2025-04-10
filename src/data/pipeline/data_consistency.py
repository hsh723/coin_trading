import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ConsistencyCheck:
    is_consistent: bool
    inconsistencies: List[Dict]
    check_timestamp: float
    severity_level: str

class DataConsistencyChecker:
    def __init__(self, check_rules: Dict):
        self.check_rules = check_rules
        
    async def check_consistency(self, datasets: List[pd.DataFrame]) -> ConsistencyCheck:
        """데이터 일관성 검사"""
        inconsistencies = []
        
        # 타임스탬프 연속성 검사
        timestamp_issues = self._check_timestamp_continuity(datasets)
        if timestamp_issues:
            inconsistencies.extend(timestamp_issues)
            
        # 데이터 중복 검사
        duplicate_issues = self._check_duplicates(datasets)
        if duplicate_issues:
            inconsistencies.extend(duplicate_issues)
            
        return ConsistencyCheck(
            is_consistent=len(inconsistencies) == 0,
            inconsistencies=inconsistencies,
            check_timestamp=pd.Timestamp.now().timestamp(),
            severity_level=self._determine_severity(inconsistencies)
        )
