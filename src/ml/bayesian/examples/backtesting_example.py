import numpy as np
import pandas as pd
import logging
import sys
import os
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 경로 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.ml.bayesian.backtesting import BayesianBacktester

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    샘플 시계열 데이터 생성
    
    Args:
        n_samples: 샘플 수
        
    Returns:
        샘플 데이터프레임
    """
    # 기본 시계열 생성
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    trend = np.linspace(100, 200, n_samples)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    noise = np.random.normal(0, 5, n_samples)
    
    # 가격 시계열 생성
    prices = trend + seasonal + noise
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    df.set_index('date', inplace=True)
    
    return df

def main():
    # 샘플 데이터 생성
    logger.info("샘플 데이터 생성 중...")
    data = generate_sample_data()
    
    # 백테스팅 시스템 초기화
    logger.info("백테스팅 시스템 초기화 중...")
    backtester = BayesianBacktester(
        model_type="ar",
        model_params={
            "ar_order": 3,
            "seasonality": True,
            "num_seasons": 24
        },
        initial_capital=10000.0,
        transaction_cost=0.001,
        save_dir="./backtesting_results"
    )
    
    # 백테스팅 실행
    logger.info("백테스팅 실행 중...")
    results = backtester.run(
        data=data,
        train_size=0.7,
        prediction_horizon=1
    )
    
    # 결과 출력
    logger.info("\n백테스팅 결과:")
    logger.info(f"총 수익률: {results['total_return']:.4f}")
    logger.info(f"연간 수익률: {results['annual_return']:.4f}")
    logger.info(f"샤프 비율: {results['sharpe_ratio']:.4f}")
    logger.info(f"최대 낙폭: {results['max_drawdown']:.4f}")
    logger.info(f"승률: {results['win_rate']:.4f}")
    logger.info(f"거래 횟수: {results['trade_count']}")
    logger.info(f"평균 거래 비용: {results['avg_cost']:.4f}")

if __name__ == "__main__":
    main() 