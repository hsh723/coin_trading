from typing import Dict, List
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    error_count: int
    warning_count: int
    validation_details: Dict[str, List[str]]

class DataValidationManager:
    def __init__(self, validation_rules: Dict):
        self.validation_rules = validation_rules
        
    async def validate_pipeline_data(self, data: pd.DataFrame) -> ValidationResult:
        """파이프라인 데이터 검증"""
        errors = []
        warnings = []
        
        # 스키마 검증
        schema_errors = self._validate_schema(data)
        errors.extend(schema_errors)
        
        # 데이터 타입 검증
        type_errors = self._validate_data_types(data)
        errors.extend(type_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_count=len(errors),
            warning_count=len(warnings),
            validation_details={'errors': errors, 'warnings': warnings}
        )
