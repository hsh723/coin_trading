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

from src.ml.bayesian.strategy_optimizer import StrategyOptimizer

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    샘플 시계열 데이터 생성
    
    Args:
        n_samples: 샘플 수
        
    Returns:
        시계열 데이터
    """
    # 기본 시계열 생성
    t = np.arange(n_samples)
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50)
    noise = np.random.normal(0, 1, n_samples)
    
    # 가격 시계열 생성
    price = 100 + trend + seasonal + noise
    price = np.exp(price / 100)  # 지수 변환으로 더 현실적인 가격 생성
    
    # 데이터프레임 생성
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    df = pd.DataFrame({
        'price': price
    }, index=dates)
    
    return df

def main():
    # 샘플 데이터 생성
    logger.info("샘플 데이터 생성 중...")
    data = generate_sample_data(n_samples=1000)
    
    # 전략 최적화 도구 초기화
    logger.info("전략 최적화 도구 초기화 중...")
    optimizer = StrategyOptimizer(
        model_type="ar",
        initial_params={
            "ar_order": 3,
            "seasonality": True,
            "num_seasons": 24
        },
        optimization_method="differential_evolution",
        n_jobs=-1,
        save_dir="./strategy_optimization"
    )
    
    # 전략 파라미터 최적화
    logger.info("전략 파라미터 최적화 시작...")
    param_bounds = {
        "ar_order": (2, 5),
        "seasonality": (True, True),
        "num_seasons": (12, 24)
    }
    
    optimization_results = optimizer.optimize_strategy(
        data=data,
        param_bounds=param_bounds,
        metric="sharpe_ratio",
        n_splits=5,
        max_iter=100
    )
    
    logger.info(f"\n최적 파라미터: {optimization_results['best_params']}")
    logger.info(f"최적 점수: {optimization_results['best_score']:.4f}")
    
    # 전략 성능 평가
    logger.info("\n전략 성능 평가 시작...")
    evaluation_results = optimizer.evaluate_strategy(
        data=data,
        params=optimization_results['best_params'],
        n_splits=5
    )
    
    logger.info("\n성능 평가 결과:")
    for metric, stats in evaluation_results.items():
        logger.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # 전략 조합 최적화
    logger.info("\n전략 조합 최적화 시작...")
    strategies = [
        {
            'model_type': 'ar',
            'params': {'ar_order': 2, 'seasonality': True, 'num_seasons': 12}
        },
        {
            'model_type': 'ar',
            'params': {'ar_order': 3, 'seasonality': True, 'num_seasons': 24}
        },
        {
            'model_type': 'ar',
            'params': {'ar_order': 4, 'seasonality': False}
        }
    ]
    
    combination_results = optimizer.optimize_strategy_combination(
        data=data,
        strategies=strategies,
        metric="sharpe_ratio"
    )
    
    logger.info("\n전략 조합 결과:")
    for i, (strategy, weight) in enumerate(zip(combination_results['strategies'], combination_results['weights'])):
        logger.info(f"전략 {i+1}: {strategy['params']}, 가중치: {weight:.4f}")
    logger.info(f"조합 점수: {combination_results['score']:.4f}")

if __name__ == "__main__":
    main() 