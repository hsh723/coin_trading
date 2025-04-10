from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]

class DataValidator:
    def __init__(self, validation_rules: Dict = None):
        self.validation_rules = validation_rules or {
            'missing_threshold': 0.1,
            'outlier_std': 3.0,
            'min_samples': 1000
        }
        
    async def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """데이터 유효성 검증"""
        errors = []
        warnings = []
        metrics = {}
        
        if len(data) < self.validation_rules['min_samples']:
            errors.append(f"Insufficient samples: {len(data)}")
            
        missing_ratio = data.isnull().sum() / len(data)
        if (missing_ratio > self.validation_rules['missing_threshold']).any():
            warnings.append("High missing value ratio detected")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=self._calculate_validation_metrics(data)
        )
