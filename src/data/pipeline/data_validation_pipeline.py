from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class ValidationPipelineResult:
    stage_results: Dict[str, bool]
    validation_time: float
    failed_checks: List[str]
    data_quality_score: float

class DataValidationPipeline:
    def __init__(self, validation_stages: List[Dict]):
        self.stages = validation_stages
        self.validation_history = []
        
    async def run_validation(self, data: pd.DataFrame) -> ValidationPipelineResult:
        """데이터 검증 파이프라인 실행"""
        start_time = time.time()
        stage_results = {}
        failed_checks = []
        
        for stage in self.stages:
            try:
                result = await self._execute_validation_stage(stage, data)
                stage_results[stage['name']] = result
                if not result:
                    failed_checks.append(stage['name'])
            except Exception as e:
                await self._handle_validation_error(e, stage['name'])
                
        return ValidationPipelineResult(
            stage_results=stage_results,
            validation_time=time.time() - start_time,
            failed_checks=failed_checks,
            data_quality_score=self._calculate_quality_score(stage_results)
        )
