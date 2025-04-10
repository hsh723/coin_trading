from typing import Dict, Any
from dataclasses import dataclass
import jsonschema

@dataclass
class ValidationError:
    field: str
    message: str
    value: Any

class SchemaValidator:
    def __init__(self, schemas: Dict[str, Dict]):
        self.schemas = schemas
        
    async def validate_data(self, data: Dict, schema_name: str) -> List[ValidationError]:
        """데이터 스키마 검증"""
        errors = []
        schema = self.schemas.get(schema_name)
        
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            errors.append(ValidationError(
                field=e.path[-1] if e.path else '',
                message=e.message,
                value=e.instance
            ))
            
        return errors
