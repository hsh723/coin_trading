from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    risk_score: float
    validation_details: List[str]
    warnings: List[str]

class StrategyValidator:
    def __init__(self, validation_rules: Dict = None):
        self.rules = validation_rules or {
            'max_position_size': 0.1,
            'max_leverage': 5.0,
            'min_profit_factor': 1.5
        }
        
    async def validate_strategy(self, strategy_config: Dict) -> ValidationResult:
        """전략 설정 검증"""
        validation_details = []
        warnings = []
        
        # 위험도 평가
        risk_score = self._evaluate_risk_score(strategy_config)
        
        # 레버리지 검증
        if strategy_config.get('leverage', 1.0) > self.rules['max_leverage']:
            validation_details.append("Leverage exceeds maximum allowed")
            
        # 포지션 크기 검증
        if strategy_config.get('position_size', 0.0) > self.rules['max_position_size']:
            validation_details.append("Position size exceeds maximum allowed")
            
        return ValidationResult(
            is_valid=len(validation_details) == 0,
            risk_score=risk_score,
            validation_details=validation_details,
            warnings=warnings
        )
