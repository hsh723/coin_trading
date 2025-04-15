"""
실행 시스템 로깅 모듈
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import aiofiles
import aiofiles.os
import pandas as pd

class ExecutionLogger:
    """실행 시스템 로거"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        실행 로거 초기화
        
        Args:
            config (Dict[str, Any]): 로깅 설정
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 로그 레벨 설정
        log_level = config.get('level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level))
        
        # 파일 핸들러 설정
        log_file = config.get('file', 'execution.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
    async def initialize(self):
        """로거 초기화"""
        self.logger.info("실행 로거 초기화 완료")
        
    async def log_execution(self, data: Dict[str, Any]):
        """
        실행 정보 로깅
        
        Args:
            data (Dict[str, Any]): 로깅할 데이터
        """
        self.logger.info(f"실행 정보: {data}")
        
    async def log_error(self, error: str, details: Dict[str, Any] = None):
        """
        오류 로깅
        
        Args:
            error (str): 오류 메시지
            details (Dict[str, Any], optional): 추가 정보
        """
        if details:
            self.logger.error(f"오류 발생: {error}, 상세 정보: {details}")
        else:
            self.logger.error(f"오류 발생: {error}")
            
    async def log_performance(self, metrics: Dict[str, Any]):
        """
        성능 메트릭 로깅
        
        Args:
            metrics (Dict[str, Any]): 성능 메트릭
        """
        self.logger.info(f"성능 메트릭: {metrics}")
        
    async def close(self):
        """로거 종료"""
        self.logger.info("실행 로거 종료")
            
    async def _create_log_directory(self):
        """로그 디렉토리 생성"""
        try:
            # 디렉토리가 없으면 생성
            if not await aiofiles.os.path.exists(self.log_dir):
                await aiofiles.os.makedirs(self.log_dir)
                
        except Exception as e:
            print(f"로그 디렉토리 생성 실패: {str(e)}")
            raise
            
    def _setup_logger(self):
        """로거 설정"""
        try:
            # 로그 레벨 설정
            level = getattr(logging, self.log_level.upper())
            self.logger.setLevel(level)
            
            # 포맷터 설정
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # 파일 핸들러 설정
            file_handler = logging.FileHandler(self.execution_log)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # 콘솔 핸들러 설정
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"로거 설정 실패: {str(e)}")
            raise
            
    async def _check_log_rotation(
        self,
        log_file: Path
    ) -> None:
        """
        로그 파일 순환 확인
        
        Args:
            log_file (Path): 로그 파일 경로
        """
        try:
            # 파일 크기 확인
            if await aiofiles.os.path.exists(log_file):
                stats = await aiofiles.os.stat(log_file)
                if stats.st_size > self.max_log_size:
                    await self._rotate_log(log_file)
                    
        except Exception as e:
            self.logger.error(f"로그 순환 확인 실패: {str(e)}")
            
    async def _rotate_log(
        self,
        log_file: Path
    ) -> None:
        """
        로그 파일 순환
        
        Args:
            log_file (Path): 로그 파일 경로
        """
        try:
            # 기존 백업 파일 순환
            for i in range(self.backup_count - 1, 0, -1):
                src = log_file.with_suffix(f'.log.{i}')
                dst = log_file.with_suffix(f'.log.{i + 1}')
                
                if await aiofiles.os.path.exists(src):
                    if await aiofiles.os.path.exists(dst):
                        await aiofiles.os.remove(dst)
                    await aiofiles.os.rename(src, dst)
                    
            # 현재 로그 파일 백업
            backup = log_file.with_suffix('.log.1')
            if await aiofiles.os.path.exists(backup):
                await aiofiles.os.remove(backup)
            await aiofiles.os.rename(log_file, backup)
            
            # 새 로그 파일 생성
            async with aiofiles.open(log_file, 'w') as f:
                await f.write('')
                
        except Exception as e:
            self.logger.error(f"로그 순환 실패: {str(e)}")
            
    async def get_execution_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        실행 로그 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 실행 로그
        """
        try:
            logs = []
            
            # 로그 파일 읽기
            async with aiofiles.open(self.execution_log, 'r') as f:
                async for line in f:
                    log_entry = json.loads(line)
                    timestamp = datetime.fromisoformat(log_entry['timestamp'])
                    
                    # 시간 범위 필터링
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                        
                    logs.append(log_entry)
                    
            # DataFrame 변환
            return pd.DataFrame(logs)
            
        except Exception as e:
            self.logger.error(f"실행 로그 조회 실패: {str(e)}")
            return pd.DataFrame()
            
    async def get_performance_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        성능 로그 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 성능 로그
        """
        try:
            logs = []
            
            # 로그 파일 읽기
            async with aiofiles.open(self.performance_log, 'r') as f:
                async for line in f:
                    log_entry = json.loads(line)
                    timestamp = datetime.fromisoformat(log_entry['timestamp'])
                    
                    # 시간 범위 필터링
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                        
                    logs.append(log_entry)
                    
            # DataFrame 변환
            return pd.DataFrame(logs)
            
        except Exception as e:
            self.logger.error(f"성능 로그 조회 실패: {str(e)}")
            return pd.DataFrame()
            
    async def get_error_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        오류 로그 조회
        
        Args:
            start_time (Optional[datetime]): 시작 시간
            end_time (Optional[datetime]): 종료 시간
            
        Returns:
            pd.DataFrame: 오류 로그
        """
        try:
            logs = []
            
            # 로그 파일 읽기
            async with aiofiles.open(self.error_log, 'r') as f:
                async for line in f:
                    log_entry = json.loads(line)
                    timestamp = datetime.fromisoformat(log_entry['timestamp'])
                    
                    # 시간 범위 필터링
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                        
                    logs.append(log_entry)
                    
            # DataFrame 변환
            return pd.DataFrame(logs)
            
        except Exception as e:
            self.logger.error(f"오류 로그 조회 실패: {str(e)}")
            return pd.DataFrame()

    def info(self, message):
        """정보 로그 기록"""
        self.logger.info(message)

    def error(self, message):
        """에러 로그 기록"""
        self.logger.error(message)

    def warning(self, message):
        """경고 로그 기록"""
        self.logger.warning(message)

    def debug(self, message):
        """디버그 로그 기록"""
        self.logger.debug(message) 