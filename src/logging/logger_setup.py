import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

class LoggerSetup:
    """로깅 시스템 설정 클래스"""
    
    def __init__(self, log_level: str = 'INFO', log_file: str = 'trading.log'):
        self.log_level = log_level
        self.log_file = log_file
        self.logger = logging.getLogger('trading')
        
    def setup(self) -> None:
        """로깅 시스템 설정"""
        try:
            # 로그 레벨 설정
            level = getattr(logging, self.log_level.upper())
            self.logger.setLevel(level)
            
            # 로그 포맷 설정
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # 콘솔 핸들러 설정
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 파일 핸들러 설정
            if self.log_file:
                # 로그 디렉토리 생성
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                    
                # 파일 핸들러 설정 (최대 10MB, 5개 파일 보관)
                file_handler = RotatingFileHandler(
                    self.log_file,
                    maxBytes=10*1024*1024,
                    backupCount=5
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
            self.logger.info("로깅 시스템 설정 완료")
            
        except Exception as e:
            print(f"로깅 시스템 설정 중 오류 발생: {str(e)}")
            raise
            
    def get_logger(self) -> logging.Logger:
        """로거 인스턴스 반환"""
        return self.logger 