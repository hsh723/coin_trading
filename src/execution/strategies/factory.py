"""
실행 전략 팩토리
"""

from typing import Dict, Any, Type
from .base import ExecutionStrategy

class ExecutionStrategyFactory:
    def __init__(self, config: Dict[str, Any]):
        """
        전략 팩토리 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self.strategies = {}
        
    def register_strategy(self, name: str, strategy_class: Type[ExecutionStrategy]):
        """
        전략 등록
        
        Args:
            name (str): 전략 이름
            strategy_class (Type[ExecutionStrategy]): 전략 클래스
        """
        self.strategies[name] = strategy_class
        
    def create_strategy(self, name: str) -> ExecutionStrategy:
        """
        전략 생성
        
        Args:
            name (str): 전략 이름
            
        Returns:
            ExecutionStrategy: 생성된 전략 인스턴스
        """
        if name not in self.strategies:
            raise ValueError(f"등록되지 않은 전략: {name}")
            
        strategy_class = self.strategies[name]
        return strategy_class(self.config)
        
    async def cleanup(self):
        """리소스 정리"""
        self.strategies.clear() 