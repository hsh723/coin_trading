"""
로깅 모듈
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any

class TradingLogger:
    """
    거래 시스템 로거 클래스
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        로거 초기화
        
        Args:
            log_dir (str): 로그 파일 저장 디렉토리
        """
        self.log_dir = log_dir
        self._setup_log_directory()
        self._setup_loggers()
    
    def _setup_log_directory(self) -> None:
        """
        로그 디렉토리 설정
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def _setup_loggers(self) -> None:
        """
        로거 설정
        """
        # 기본 로거 설정
        self.logger = logging.getLogger("trading_system")
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러 설정 (일별 롤오버)
        daily_handler = RotatingFileHandler(
            filename=os.path.join(self.log_dir, f"trading_{datetime.now().strftime('%Y%m%d')}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=30,  # 30일 보관
            encoding='utf-8'
        )
        daily_handler.setLevel(logging.INFO)
        
        # 에러 로그 핸들러 설정
        error_handler = RotatingFileHandler(
            filename=os.path.join(self.log_dir, "error.log"),
            maxBytes=10*1024*1024,
            backupCount=30,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # 거래 로그 핸들러 설정
        trade_handler = RotatingFileHandler(
            filename=os.path.join(self.log_dir, "trades.log"),
            maxBytes=10*1024*1024,
            backupCount=30,
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        default_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'File: %(pathname)s:%(lineno)d\n'
            'Function: %(funcName)s\n'
            'Exception: %(exc_info)s'
        )
        
        trade_formatter = logging.Formatter(
            '%(asctime)s - TRADE - %(message)s'
        )
        
        # 핸들러에 포맷터 설정
        daily_handler.setFormatter(default_formatter)
        error_handler.setFormatter(error_formatter)
        trade_handler.setFormatter(trade_formatter)
        console_handler.setFormatter(default_formatter)
        
        # 핸들러 추가
        self.logger.addHandler(daily_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(trade_handler)
        self.logger.addHandler(console_handler)
        
        # 거래 로거 설정
        self.trade_logger = logging.getLogger("trading_system.trades")
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.addHandler(trade_handler)
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        order_type: str,
        order_id: Optional[str] = None,
        status: str = "executed"
    ) -> None:
        """
        거래 로그 기록
        
        Args:
            symbol (str): 거래 심볼
            side (str): 포지션 방향
            amount (float): 주문 수량
            price (float): 주문 가격
            order_type (str): 주문 유형
            order_id (str, optional): 주문 ID
            status (str): 주문 상태
        """
        trade_info = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "order_type": order_type,
            "order_id": order_id,
            "status": status
        }
        
        self.trade_logger.info(f"TRADE: {trade_info}")
    
    def log_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        unrealized_pnl: float,
        leverage: float
    ) -> None:
        """
        포지션 로그 기록
        
        Args:
            symbol (str): 거래 심볼
            side (str): 포지션 방향
            size (float): 포지션 크기
            entry_price (float): 진입 가격
            unrealized_pnl (float): 미실현 손익
            leverage (float): 레버리지
        """
        position_info = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "unrealized_pnl": unrealized_pnl,
            "leverage": leverage
        }
        
        self.trade_logger.info(f"POSITION: {position_info}")
    
    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        오류 로그 기록
        
        Args:
            error (Exception): 오류 객체
            context (str, optional): 오류 발생 컨텍스트
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self.logger.error(f"ERROR: {error_info}", exc_info=True)
    
    def log_system_status(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        시스템 상태 로그 기록
        
        Args:
            status (str): 시스템 상태
            details (dict, optional): 상세 정보
        """
        status_info = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details or {}
        }
        
        self.logger.info(f"SYSTEM_STATUS: {status_info}")
    
    def log_performance_metrics(
        self,
        metrics: Dict[str, Any],
        period: str = "daily"
    ) -> None:
        """
        성과 지표 로그 기록
        
        Args:
            metrics (dict): 성과 지표
            period (str): 기간 (daily/weekly/monthly)
        """
        metrics_info = {
            "timestamp": datetime.now().isoformat(),
            "period": period,
            "metrics": metrics
        }
        
        self.logger.info(f"PERFORMANCE: {metrics_info}")
    
    def log_strategy_signal(
        self,
        symbol: str,
        signal_type: str,
        indicators: Dict[str, Any],
        conditions: Dict[str, bool]
    ) -> None:
        """
        전략 신호 로그 기록
        
        Args:
            symbol (str): 거래 심볼
            signal_type (str): 신호 유형
            indicators (dict): 지표 값
            conditions (dict): 조건 충족 여부
        """
        signal_info = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal_type": signal_type,
            "indicators": indicators,
            "conditions": conditions
        }
        
        self.logger.info(f"STRATEGY_SIGNAL: {signal_info}") 