from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VerificationResult:
    is_valid: bool
    errors: List[str]
    validation_rules: List[str]
    metadata: Dict

class TaskVerifier:
    def __init__(self, verification_rules: Dict):
        self.rules = verification_rules
        
    async def verify_task(self, task: Dict) -> VerificationResult:
        """작업 검증 수행"""
        errors = []
        applied_rules = []
        
        for rule_name, rule_func in self.rules.items():
            try:
                if not await rule_func(task):
                    errors.append(f"Failed rule: {rule_name}")
                applied_rules.append(rule_name)
            except Exception as e:
                errors.append(f"Rule error {rule_name}: {str(e)}")
                
        return VerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            validation_rules=applied_rules,
            metadata={'task_id': task.get('id')}
        )
