import numpy as np
import pandas as pd
import logging
import sys
import os
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 부모 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.bayesian.risk_management import RiskManager

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
    
    # 리스크 관리자 초기화
    logger.info("리스크 관리자 초기화 중...")
    risk_manager = RiskManager(
        initial_capital=10000.0,
        max_position_size=0.1,
        max_drawdown=0.2,
        var_confidence=0.95,
        save_dir="./risk_management_results"
    )
    
    # 포지션 사이즈 계산
    logger.info("포지션 사이즈 계산 중...")
    current_price = data['price'].iloc[-1]
    volatility = data['price'].pct_change().std()
    position_size = risk_manager.calculate_position_size(
        price=current_price,
        volatility=volatility
    )
    logger.info(f"계산된 포지션 사이즈: {position_size:.2f}")
    
    # VaR 계산
    logger.info("VaR 계산 중...")
    returns = data['price'].pct_change().dropna()
    var = risk_manager.calculate_var(returns)
    logger.info(f"계산된 VaR: {var:.4f}")
    
    # 스트레스 테스트 수행
    logger.info("스트레스 테스트 수행 중...")
    scenarios = [
        {'name': 'mild_shock', 'price_shock': -0.1, 'volatility_shock': 0.2},
        {'name': 'severe_shock', 'price_shock': -0.3, 'volatility_shock': 0.5}
    ]
    stress_test_results = risk_manager.perform_stress_test(data, scenarios)
    logger.info("스트레스 테스트 결과:")
    for scenario, results in stress_test_results.items():
        logger.info(f"{scenario}:")
        logger.info(f"  VaR: {results['var']:.4f}")
        logger.info(f"  최대 낙폭: {results['max_drawdown']:.4f}")
    
    # 상관관계 분석
    logger.info("상관관계 분석 중...")
    correlation_results = risk_manager.analyze_correlations(data)
    logger.info("상관관계 분석 결과:")
    logger.info(f"평균 상관관계: {correlation_results['mean_correlation']:.4f}")
    logger.info(f"최대 상관관계: {correlation_results['max_correlation']:.4f}")
    
    # 리스크 메트릭 모니터링
    logger.info("리스크 메트릭 모니터링 중...")
    portfolio_value = 10000.0
    position = 1.0
    risk_metrics = risk_manager.monitor_risk_metrics(
        portfolio_value=portfolio_value,
        position=position,
        price=current_price
    )
    logger.info("리스크 메트릭:")
    logger.info(f"낙폭: {risk_metrics['drawdown']:.4f}")
    logger.info(f"포지션 크기: {risk_metrics['position_size']:.4f}")
    if risk_metrics['warnings']:
        logger.warning("경고:")
        for warning in risk_metrics['warnings']:
            logger.warning(f"  - {warning}")

if __name__ == "__main__":
    main() 