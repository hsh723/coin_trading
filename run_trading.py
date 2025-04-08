"""
암호화폐 트레이딩 봇 실행 스크립트
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.trading.trader import Trader
from src.strategy.momentum import MomentumStrategy
from src.risk.risk_manager import RiskManager
from src.utils.logger import setup_logging

def main():
    """메인 함수"""
    # 환경 변수 로드
    load_dotenv()
    
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 트레이더 초기화
        trader = Trader(
            api_key=os.getenv('API_KEY'),
            api_secret=os.getenv('API_SECRET'),
            strategy=MomentumStrategy(),
            risk_manager=RiskManager(),
            symbols=os.getenv('SYMBOLS', 'BTC/USDT').split(','),
            exchange_id=os.getenv('EXCHANGE_ID', 'binance')
        )
        
        # 트레이딩 시작
        logger.info("트레이딩 시작")
        trader.start_trading(mode='live')
        
    except Exception as e:
        logger.error(f"트레이딩 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 