from typing import Dict
import pandas as pd
from dataclasses import dataclass

@dataclass
class TaskStatistics:
    success_rate: float
    avg_execution_time: float
    error_distribution: Dict[str, int]
    performance_trends: pd.DataFrame

class TaskStatisticsAnalyzer:
    def __init__(self):
        self.stats_history = []
        
    async def analyze_statistics(self, task_history: List[Dict]) -> TaskStatistics:
        """작업 통계 분석"""
        df = pd.DataFrame(task_history)
        
        return TaskStatistics(
            success_rate=self._calculate_success_rate(df),
            avg_execution_time=self._calculate_avg_execution_time(df),
            error_distribution=self._analyze_errors(df),
            performance_trends=self._analyze_performance_trends(df)
        )
