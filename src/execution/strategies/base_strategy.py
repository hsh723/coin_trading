"""
실행 전략 기본 모듈

모든 실행 전략의 기본 클래스를 정의합니다.
"""

import abc
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class BaseExecutionStrategy(abc.ABC):
    """실행 전략 기본 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        실행 전략 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
        """
        self.config = config
        self.name = self.__class__.__name__
        
    @abc.abstractmethod
    async def execute(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        주문 실행
        
        Args:
            order_request (Dict[str, Any]): 주문 요청 정보
            
        Returns:
            Dict[str, Any]: 실행 결과
        """
        pass
        
    @abc.abstractmethod
    async def cancel(self) -> bool:
        """
        실행 취소
        
        Returns:
            bool: 취소 성공 여부
        """
        pass
        
    def get_name(self) -> str:
        """
        전략 이름 조회
        
        Returns:
            str: 전략 이름
        """
        return self.name 