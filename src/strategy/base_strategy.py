"""
기본 전략 클래스
모든 전략의 기본이 되는 추상 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        기본 전략 초기화
        
        Args:
            config (Optional[Dict[str, Any]], optional): 전략 설정. Defaults to None.
        """
        self.config = config or {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "parameters": {}
        }
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """로깅 설정"""
        self.logger.setLevel(logging.INFO)
        
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        데이터 유효성 검사
        
        Args:
            data (pd.DataFrame): 검증할 데이터
            
        Raises:
            ValueError: 데이터가 비어있거나 필수 컬럼이 없는 경우
        """
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
        
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> None:
        """
        전략 초기화
        
        Args:
            data (pd.DataFrame): 초기화에 사용할 데이터
            
        Raises:
            ValueError: 데이터가 비어있거나 필수 컬럼이 없는 경우
        """
        self._validate_data(data)
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시장 분석"""
        pass
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        거래 신호 생성
        
        Args:
            data (pd.DataFrame): 신호 생성에 사용할 데이터
            
        Returns:
            Dict[str, Any]: 생성된 거래 신호
            
        Raises:
            ValueError: 데이터가 비어있거나 필수 컬럼이 없는 경우
        """
        self._validate_data(data)
        
    @abstractmethod
    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """거래 실행"""
        pass
        
    @abstractmethod
    def update(self, data: pd.DataFrame) -> None:
        """전략 상태 업데이트"""
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """전략 상태 조회"""
        pass
        
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """전략 상태 설정"""
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