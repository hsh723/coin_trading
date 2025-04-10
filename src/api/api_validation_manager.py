from typing import Dict, Optional
from dataclasses import dataclass
import jsonschema

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warning_messages: List[str]
    validation_context: Dict

class ApiValidationManager:
    def __init__(self, schema_config: Dict = None):
        self.config = schema_config or {
            'strict_mode': True,
            'validate_responses': True
        }
        self.schemas = {}
        
    async def validate_request(self, request_data: Dict, 
                             endpoint: str) -> ValidationResult:
        """API 요청 데이터 검증"""
        schema = self._get_request_schema(endpoint)
        validation_errors = []
        warnings = []
        
        try:
            jsonschema.validate(instance=request_data, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            validation_errors.append(str(e))
            
        return ValidationResult(
            is_valid=len(validation_errors) == 0,
            errors=validation_errors,
            warning_messages=warnings,
            validation_context={'endpoint': endpoint}
        )
