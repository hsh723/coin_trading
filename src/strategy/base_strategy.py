"""
기본 전략 클래스
모든 전략의 기본이 되는 추상 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
from src.analysis.indicators.technical import TechnicalIndicators
from src.utils.logger import get_logger

@dataclass
class StrategyResult:
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    params: Dict
    metadata: Dict

class BaseStrategy(ABC):
    """전략 기본 클래스"""
    
    def __init__(self, strategy_config: Dict = None):
        self.config = strategy_config or {}
        self.position = None
        self.last_signal = None
        self.logger = get_logger(__name__)
        self._state: Dict[str, Any] = {}
        
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> None:
        """전략 초기화"""
        pass
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시장 분석"""
        pass
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """매매 신호 생성"""
        pass
        
    @abstractmethod
    async def generate_signal(self, market_data: Dict) -> StrategyResult:
        """전략 신호 생성"""
        pass
        
    @abstractmethod
    def execute(self, data: pd.DataFrame, position: Optional[float] = None) -> Dict[str, Any]:
        """매매 실행"""
        pass
        
    @abstractmethod
    def update(self, data: pd.DataFrame) -> None:
        """전략 상태 업데이트"""
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """전략 상태 반환"""
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