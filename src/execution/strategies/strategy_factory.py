"""
전략 팩토리 모듈

다양한 실행 전략을 생성하고 관리하는 팩토리 클래스입니다.
"""

import logging
from typing import Dict, Any, List, Optional, Type

from src.execution.strategies.base_strategy import BaseExecutionStrategy
from src.execution.strategies.adaptive_strategy import AdaptiveExecutionStrategy
from src.execution.strategies.twap_strategy import TwapExecutionStrategy
from src.execution.strategies.vwap_strategy import VwapExecutionStrategy

logger = logging.getLogger(__name__)

class ExecutionStrategyFactory:
    """실행 전략 팩토리"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        실행 전략 팩토리 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self._strategies: Dict[str, Type[BaseExecutionStrategy]] = {}
        self._initialized_strategies: Dict[str, BaseExecutionStrategy] = {}
        
        # 기본 전략 등록
        self._register_default_strategies()
        
    def _register_default_strategies(self) -> None:
        """기본 전략 등록"""
        self.register_strategy('adaptive', AdaptiveExecutionStrategy)
        self.register_strategy('twap', TwapExecutionStrategy)
        self.register_strategy('vwap', VwapExecutionStrategy)
        
    def register_strategy(
        self,
        name: str,
        strategy_class: Type[BaseExecutionStrategy]
    ) -> None:
        """
        전략 등록
        
        Args:
            name (str): 전략 이름
            strategy_class (Type[BaseExecutionStrategy]): 전략 클래스
        """
        self._strategies[name] = strategy_class
        logger.info(f"전략 등록: {name} ({strategy_class.__name__})")
        
    def get_strategy(self, name: str) -> Optional[BaseExecutionStrategy]:
        """
        전략 인스턴스 조회
        
        Args:
            name (str): 전략 이름
            
        Returns:
            Optional[BaseExecutionStrategy]: 전략 인스턴스
        """
        # 이미 초기화된 전략이 있는 경우
        if name in self._initialized_strategies:
            return self._initialized_strategies[name]
            
        # 새 전략 인스턴스 생성
        if name in self._strategies:
            strategy_class = self._strategies[name]
            
            # 전략별 설정 가져오기
            strategy_config = self.config.get(name, {})
            
            # 전략 인스턴스 생성
            try:
                strategy = strategy_class(strategy_config)
                self._initialized_strategies[name] = strategy
                return strategy
            except Exception as e:
                logger.error(f"전략 초기화 실패: {name} - {str(e)}")
                
        logger.warning(f"알 수 없는 전략: {name}")
        return None
        
    def get_available_strategies(self) -> List[str]:
        """
        사용 가능한 전략 목록 조회
        
        Returns:
            List[str]: 전략 이름 목록
        """
        return list(self._strategies.keys())
        
    async def cleanup(self) -> None:
        """리소스 정리"""
        # 모든 초기화된 전략의 리소스 정리
        for name, strategy in self._initialized_strategies.items():
            try:
                await strategy.cancel()
            except Exception as e:
                logger.error(f"전략 정리 실패: {name} - {str(e)}")
                
        self._initialized_strategies.clear()
        logger.info("전략 팩토리 정리 완료") 