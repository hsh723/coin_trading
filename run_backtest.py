"""
암호화폐 트레이딩 봇 백테스팅 실행 스크립트
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from src.backtesting.engine import BacktestingEngine
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
        # 백테스팅 엔진 초기화
        engine = BacktestingEngine(
            strategy=MomentumStrategy(),
            risk_manager=RiskManager(),
            initial_capital=float(os.getenv('INITIAL_CAPITAL', '10000')),
            commission=float(os.getenv('COMMISSION', '0.001'))
        )
        
        # 백테스팅 설정
        symbol = os.getenv('SYMBOL', 'BTC/USDT')
        timeframe = os.getenv('TIMEFRAME', '1h')
        start_date = datetime.now() - timedelta(days=int(os.getenv('BACKTEST_DAYS', '30')))
        end_date = datetime.now()
        
        # 백테스팅 실행
        logger.info("백테스팅 시작")
        results = engine.run(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # 결과 저장
        results_path = Path('results/backtest')
        results_path.mkdir(exist_ok=True)
        results.to_csv(results_path / f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        logger.info("백테스팅 완료")
        
    except Exception as e:
        logger.error(f"백테스팅 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 