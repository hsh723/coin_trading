from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    error_messages: List[str]
    warnings: List[str]
    metadata: Dict

class ExecutionValidator:
    def __init__(self, validation_rules: Dict = None):
        self.rules = validation_rules or {
            'min_order_size': 0.001,
            'max_order_size': 100.0,
            'price_deviation': 0.05,  # 5%
            'min_margin_ratio': 0.05  # 5%
        }
        
    async def validate_execution(self, order: Dict, 
                               market_data: Dict) -> ValidationResult:
        """실행 유효성 검증"""
        errors = []
        warnings = []
        
        # 주문 크기 검증
        if not self._validate_order_size(order, errors):
            return self._create_invalid_result(errors)
            
        # 가격 검증
        if not self._validate_price(order, market_data, errors, warnings):
            return self._create_invalid_result(errors)
            
        # 마진 검증
        if not self._validate_margin(order, warnings):
            warnings.append("Low margin ratio detected")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            error_messages=errors,
            warnings=warnings,
            metadata={'order_id': order.get('id')}
        )
