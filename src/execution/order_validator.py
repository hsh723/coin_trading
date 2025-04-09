from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class OrderValidator:
    def __init__(self, limits: Dict = None):
        self.limits = limits or {
            'min_order_size': 0.001,
            'max_order_size': 100.0,
            'price_deviation': 0.05
        }
        
    def validate_order(self, order: Dict, market_data: Dict) -> ValidationResult:
        """주문 유효성 검증"""
        errors = []
        warnings = []
        
        self._validate_size(order, errors, warnings)
        self._validate_price(order, market_data, errors, warnings)
        self._validate_balance(order, errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
