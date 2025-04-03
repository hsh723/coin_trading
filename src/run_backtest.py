"""
백테스트 실행 스크립트
"""

import os
import sys
import argparse
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config_loader import get_config
from utils.logger import setup_logger
from data.collector import DataCollector
from data.processor import DataProcessor
from strategies.bollinger_rsi import BollingerRSIStrategy
from backtest.engine import Backtester
from backtest.evaluator import StrategyEvaluator

class BacktestRunner:
    """
    백테스트 실행 클래스
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        백테스트 실행기 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # 설정 로드
        self.config = get_config(config_path)
        
        # 로거 초기화
        self.logger = setup_logger()
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        self.logger.info("Backtest runner initialized")
    
    def _initialize_components(self) -> None:
        """
        컴포넌트 초기화
        """
        try:
            # 데이터 수집기 초기화
            self.collector = DataCollector(
                exchange=self.config['exchange'],
                symbols=self.config['trading']['symbols'],
                timeframes=self.config['trading']['timeframes']
            )
            
            # 데이터 처리기 초기화
            self.processor = DataProcessor(
                config=self.config['data_processing']
            )
            
            # 전략 초기화
            strategy_config = self.config['strategies']['bollinger_rsi']
            self.strategy = BollingerRSIStrategy(
                symbol=strategy_config['symbol'],
                timeframe=strategy_config['timeframe'],
                initial_capital=strategy_config['initial_capital'],
                position_size=strategy_config['position_size'],
                stop_loss=strategy_config['stop_loss'],
                take_profit=strategy_config['take_profit']
            )
            
            # 백테스트 엔진 초기화
            self.backtester = Backtester(
                strategy=self.strategy,
                initial_capital=strategy_config['initial_capital'],
                commission=self.config['trading']['commission'],
                slippage=self.config['trading']['slippage']
            )
            
            # 평가기 초기화
            self.evaluator = StrategyEvaluator(
                save_dir=self.config['backtest']['results_dir']
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    async def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str,
        timeframe: str,
        initial_capital: float,
        position_size: float,
        stop_loss: float,
        take_profit: float
    ) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            start_date (datetime): 시작 날짜
            end_date (datetime): 종료 날짜
            symbol (str): 거래 심볼
            timeframe (str): 시간 프레임
            initial_capital (float): 초기 자본금
            position_size (float): 포지션 크기
            stop_loss (float): 손절 비율
            take_profit (float): 익절 비율
            
        Returns:
            Dict[str, Any]: 백테스트 결과
        """
        try:
            # 전략 파라미터 업데이트
            self.strategy.update_parameters(
                symbol=symbol,
                timeframe=timeframe,
                initial_capital=initial_capital,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # 데이터 수집
            self.logger.info(f"Collecting data for {symbol} {timeframe}")
            data = await self.collector.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            # 데이터 전처리
            self.logger.info("Processing data")
            processed_data = await self.processor.process_data(data)
            
            # 백테스트 실행
            self.logger.info("Running backtest")
            results = await self.backtester.run(
                data=processed_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # 결과 평가
            self.logger.info("Evaluating results")
            evaluation_results = self.evaluator.evaluate(results)
            
            # 결과 시각화
            self.logger.info("Generating visualizations")
            self.evaluator.plot_results()
            
            # 결과 저장
            self.logger.info("Saving results")
            self._save_results(results, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _save_results(
        self,
        backtest_results: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> None:
        """
        결과 저장
        
        Args:
            backtest_results (dict): 백테스트 결과
            evaluation_results (dict): 평가 결과
        """
        try:
            # 결과 디렉토리 생성
            results_dir = self.config['backtest']['results_dir']
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 백테스트 결과 저장
            backtest_file = os.path.join(results_dir, f"backtest_results_{timestamp}.json")
            with open(backtest_file, 'w') as f:
                pd.DataFrame(backtest_results).to_json(f, orient='records')
            
            # 평가 결과 저장
            evaluation_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
            with open(evaluation_file, 'w') as f:
                pd.DataFrame([evaluation_results]).to_json(f, orient='records')
            
            self.logger.info(f"Results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

def parse_args() -> argparse.Namespace:
    """
    명령행 인자 파싱
    
    Returns:
        argparse.Namespace: 파싱된 인자
    """
    parser = argparse.ArgumentParser(description="Cryptocurrency Backtesting System")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading symbol"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Trading timeframe"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital"
    )
    
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.1,
        help="Position size (as fraction of capital)"
    )
    
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop loss ratio"
    )
    
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.04,
        help="Take profit ratio"
    )
    
    return parser.parse_args()

async def main() -> None:
    """
    메인 실행 함수
    """
    # 명령행 인자 파싱
    args = parse_args()
    
    try:
        # 날짜 파싱
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # 백테스트 실행기 초기화
        runner = BacktestRunner(config_path=args.config)
        
        # 백테스트 실행
        results = await runner.run_backtest(
            start_date=start_date,
            end_date=end_date,
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_capital=args.initial_capital,
            position_size=args.position_size,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit
        )
        
        # 결과 출력
        print("\nBacktest Results:")
        print(f"Total Return: {results['performance_metrics']['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['risk_metrics']['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['trade_statistics']['total_trades']}")
        print(f"Win Rate: {results['performance_metrics']['win_rate']:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main()) 