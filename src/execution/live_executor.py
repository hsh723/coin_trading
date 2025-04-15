import asyncio
import logging
from typing import Dict, Any
from config.env_loader import EnvLoader
from logging.logger_setup import LoggerSetup
from auth.api_key_manager import APIKeyManager
from execution.execution_manager import ExecutionManager

class LiveExecutor:
    """실시간 거래 실행 진입점"""
    
    def __init__(self):
        self.env_loader = EnvLoader()
        self.logger_setup = LoggerSetup()
        self.api_key_manager = APIKeyManager()
        self.execution_manager = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """초기화"""
        try:
            # 환경 변수 로드
            self.env_loader.load()
            
            # 로깅 시스템 설정
            self.logger_setup.setup()
            
            # API 키 관리자 초기화
            self.api_key_manager.initialize()
            
            # 실행 관리자 초기화
            self.execution_manager = ExecutionManager()
            await self.execution_manager.initialize()
            
            self.logger.info("실시간 거래 실행기 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"실시간 거래 실행기 초기화 중 오류 발생: {str(e)}")
            raise
            
    async def execute_trade(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """거래 실행"""
        try:
            if not self.execution_manager:
                raise RuntimeError("실행 관리자가 초기화되지 않았습니다.")
                
            # 거래 실행
            result = await self.execution_manager.execute_order(order)
            
            self.logger.info(f"거래 실행 완료: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류 발생: {str(e)}")
            raise
            
    async def close(self) -> None:
        """종료"""
        try:
            if self.execution_manager:
                await self.execution_manager.close()
                
            self.logger.info("실시간 거래 실행기 종료 완료")
            
        except Exception as e:
            self.logger.error(f"실시간 거래 실행기 종료 중 오류 발생: {str(e)}")
            raise 