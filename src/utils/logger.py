"""
로깅 모듈
로깅 설정 및 관리를 위한 기능을 제공합니다.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
from ..utils.database import DatabaseManager

def setup_logger(name: str = "coin_trading") -> logging.Logger:
    """
    로거 설정
    
    Args:
        name (str): 로거 이름
        
    Returns:
        logging.Logger: 설정된 로거
    """
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일 경로 설정
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    로거 조회
    
    Args:
        name (Optional[str]): 로거 이름
        
    Returns:
        logging.Logger: 로거
    """
    if name is None:
        name = "coin_trading"
    return logging.getLogger(name)

def log_message(logger: logging.Logger, message: str, level: str = "info") -> None:
    """
    로그 메시지 기록
    
    Args:
        logger (logging.Logger): 로거
        message (str): 로그 메시지
        level (str): 로그 레벨
    """
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "critical":
        logger.critical(message)

def clear_old_logs(days: int = 30) -> None:
    """
    오래된 로그 파일 삭제
    
    Args:
        days (int): 보관 기간 (일)
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        return
        
    current_time = datetime.now()
    for log_file in log_dir.glob("*.log"):
        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
        if (current_time - file_time).days > days:
            log_file.unlink()

class TradeLogger:
    """거래 로깅 클래스"""
    
    def __init__(self, db: DatabaseManager):
        """
        초기화
        
        Args:
            db (DatabaseManager): 데이터베이스 관리자
        """
        self.db = db
        self.logger = setup_logger(__name__)
        
    async def log_trade_entry(self, trade: Dict[str, Any]):
        """
        거래 진입 로깅
        
        Args:
            trade (Dict[str, Any]): 거래 정보
        """
        try:
            # 로그 메시지 생성
            message = (
                f"진입 - 심볼: {trade['symbol']}, "
                f"방향: {trade['side']}, "
                f"가격: {trade['price']:.2f}, "
                f"크기: {trade['size']:.4f}, "
                f"이유: {trade.get('reason', 'N/A')}"
            )
            
            # 로그 저장
            self.logger.info(message)
            await self.db.save_log('INFO', message, 'trade')
            
            # 거래 기록 저장
            await self.db.save_trade(trade)
            
        except Exception as e:
            self.logger.error(f"거래 진입 로깅 실패: {str(e)}")
            
    async def log_trade_exit(self, trade: Dict[str, Any]):
        """
        거래 청산 로깅
        
        Args:
            trade (Dict[str, Any]): 거래 정보
        """
        try:
            # 로그 메시지 생성
            message = (
                f"청산 - 심볼: {trade['symbol']}, "
                f"방향: {trade['side']}, "
                f"가격: {trade['price']:.2f}, "
                f"크기: {trade['size']:.4f}, "
                f"손익: {trade.get('pnl', 0):.2f}, "
                f"이유: {trade.get('reason', 'N/A')}"
            )
            
            # 로그 저장
            self.logger.info(message)
            await self.db.save_log('INFO', message, 'trade')
            
            # 거래 기록 업데이트
            await self.db.update_trade(trade['id'], trade)
            
        except Exception as e:
            self.logger.error(f"거래 청산 로깅 실패: {str(e)}")
            
    async def log_position_update(self, position: Dict[str, Any]):
        """
        포지션 업데이트 로깅
        
        Args:
            position (Dict[str, Any]): 포지션 정보
        """
        try:
            # 로그 메시지 생성
            message = (
                f"포지션 업데이트 - 심볼: {position['symbol']}, "
                f"방향: {position['side']}, "
                f"진입가: {position['entry_price']:.2f}, "
                f"현재가: {position['current_price']:.2f}, "
                f"크기: {position['size']:.4f}, "
                f"손익: {position.get('pnl', 0):.2f}"
            )
            
            # 로그 저장
            self.logger.info(message)
            await self.db.save_log('INFO', message, 'position')
            
            # 포지션 기록 업데이트
            await self.db.update_position(position['id'], position)
            
        except Exception as e:
            self.logger.error(f"포지션 업데이트 로깅 실패: {str(e)}")
            
    async def log_error(self, error: Exception, context: str = ''):
        """
        에러 로깅
        
        Args:
            error (Exception): 에러 객체
            context (str): 에러 컨텍스트
        """
        try:
            # 로그 메시지 생성
            message = f"에러 발생 - 컨텍스트: {context}, 메시지: {str(error)}"
            
            # 로그 저장
            self.logger.error(message)
            await self.db.save_log('ERROR', message, 'system')
            
        except Exception as e:
            print(f"에러 로깅 실패: {str(e)}")
            
    async def log_performance(self, performance: Dict[str, Any]):
        """
        성과 로깅
        
        Args:
            performance (Dict[str, Any]): 성과 정보
        """
        try:
            # 로그 메시지 생성
            message = (
                f"성과 업데이트 - "
                f"자본: {performance['capital']:.2f}, "
                f"수익률: {performance['returns']:.2%}, "
                f"승률: {performance.get('win_rate', 0):.2%}, "
                f"샤프 비율: {performance.get('sharpe_ratio', 0):.2f}"
            )
            
            # 로그 저장
            self.logger.info(message)
            await self.db.save_log('INFO', message, 'performance')
            
            # 성과 기록 저장
            await self.db.save_performance(performance)
            
        except Exception as e:
            self.logger.error(f"성과 로깅 실패: {str(e)}")
            
    async def get_trade_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list:
        """
        거래 로그 조회
        
        Args:
            start_date (Optional[datetime]): 시작 일자
            end_date (Optional[datetime]): 종료 일자
            
        Returns:
            list: 거래 로그
        """
        try:
            return await self.db.get_logs('trade', start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"거래 로그 조회 실패: {str(e)}")
            return []
            
    async def get_error_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list:
        """
        에러 로그 조회
        
        Args:
            start_date (Optional[datetime]): 시작 일자
            end_date (Optional[datetime]): 종료 일자
            
        Returns:
            list: 에러 로그
        """
        try:
            return await self.db.get_logs('ERROR', start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"에러 로그 조회 실패: {str(e)}")
            return []
            
    async def export_logs(self, file_path: str):
        """
        로그 내보내기
        
        Args:
            file_path (str): 파일 경로
        """
        try:
            # 모든 로그 조회
            logs = await self.db.get_all_logs()
            
            # JSON 파일로 저장
            with open(file_path, 'w') as f:
                json.dump(logs, f, indent=4, default=str)
                
        except Exception as e:
            self.logger.error(f"로그 내보내기 실패: {str(e)}") 