"""
기본 전략 클래스
모든 전략의 기본이 되는 추상 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class BaseStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        """
        기본 전략 초기화
        
        Args:
            config (Dict[str, Any]): 전략 설정
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """로깅 설정"""
        self.logger.setLevel(logging.INFO)
        
    @abstractmethod
    def initialize(self) -> None:
        """전략 초기화"""
        pass
        
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 분석
        
        Args:
            data (Dict[str, Any]): 분석할 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        pass
        
    @abstractmethod
    def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        거래 신호 생성
        
        Args:
            analysis (Dict[str, Any]): 분석 결과
            
        Returns:
            Dict[str, Any]: 거래 신호
        """
        pass
        
    @abstractmethod
    def execute(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        거래 실행
        
        Args:
            signals (Dict[str, Any]): 거래 신호
            
        Returns:
            Dict[str, Any]: 거래 결과
        """
        pass
        
    @abstractmethod
    def update(self, data: Dict[str, Any]) -> None:
        """
        전략 업데이트
        
        Args:
            data (Dict[str, Any]): 업데이트할 데이터
        """
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        전략 상태 조회
        
        Returns:
            Dict[str, Any]: 전략 상태
        """
        pass
        
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        전략 상태 설정
        
        Args:
            state (Dict[str, Any]): 설정할 상태
        """
        pass
        
    def log(self, message: str, level: str = "info") -> None:
        """
        로그 기록
        
        Args:
            message (str): 로그 메시지
            level (str): 로그 레벨
        """
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
            
    def get_timestamp(self) -> str:
        """
        현재 타임스탬프 조회
        
        Returns:
            str: 현재 타임스탬프
        """
        return datetime.now().isoformat()
        
    def validate_config(self) -> bool:
        """
        설정 검증
        
        Returns:
            bool: 설정 유효성
        """
        required_keys = ["name", "version", "parameters"]
        return all(key in self.config for key in required_keys)
        
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        파라미터 조회
        
        Args:
            key (str): 파라미터 키
            default (Any): 기본값
            
        Returns:
            Any: 파라미터 값
        """
        return self.config.get("parameters", {}).get(key, default)
        
    def set_parameter(self, key: str, value: Any) -> None:
        """
        파라미터 설정
        
        Args:
            key (str): 파라미터 키
            value (Any): 파라미터 값
        """
        if "parameters" not in self.config:
            self.config["parameters"] = {}
        self.config["parameters"][key] = value 