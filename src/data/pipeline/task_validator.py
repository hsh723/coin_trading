from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    validation_errors: List[str]
    validation_warnings: List[str]
    metadata: Dict

class TaskValidator:
    def __init__(self, validation_rules: Dict = None):
        self.rules = validation_rules or {
            'required_fields': ['task_id', 'type'],
            'field_types': {
                'priority': int,
                'retry_count': int,
                'timeout': float
            }
        }
        
    async def validate_task(self, task: Dict) -> ValidationResult:
        """작업 유효성 검증"""
        errors = []
        warnings = []
        
        # 필수 필드 검증
        for field in self.rules['required_fields']:
            if field not in task:
                errors.append(f"Missing required field: {field}")
                
        # 필드 타입 검증
        for field, expected_type in self.rules['field_types'].items():
            if field in task and not isinstance(task[field], expected_type):
                errors.append(f"Invalid type for {field}: expected {expected_type}")
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            validation_errors=errors,
            validation_warnings=warnings,
            metadata={'task_id': task.get('task_id')}
        )
