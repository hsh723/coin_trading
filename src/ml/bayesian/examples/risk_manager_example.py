import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..risk_manager import RiskManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_days: int = 100) -> Dict[str, pd.Series]:
    """
    샘플 데이터 생성
    
    Args:
        n_days: 일수
        
    Returns:
        수익률 데이터 딕셔너리
    """
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # BTC 수익률 생성
    btc_returns = np.random.normal(0.001, 0.02, n_days)
    btc_returns = pd.Series(btc_returns, index=dates, name='BTC')
    
    # ETH 수익률 생성 (BTC와 상관관계 있음)
    eth_returns = 0.7 * btc_returns + np.random.normal(0.001, 0.015, n_days)
    eth_returns = pd.Series(eth_returns, index=dates, name='ETH')
    
    return {'BTC': btc_returns, 'ETH': eth_returns}

def main():
    """메인 실행 함수"""
    try:
        # 리스크 매니저 초기화
        risk_manager = RiskManager(
            initial_capital=10000.0,
            max_position_size=0.1,
            max_drawdown=0.2,
            var_confidence=0.95,
            save_dir="./risk_reports"
        )
        
        # 샘플 데이터 생성
        returns = generate_sample_data()
        
        # VaR 계산
        for symbol, data in returns.items():
            var = risk_manager.calculate_var(data)
            logger.info(f"{symbol} VaR: {var:.2%}")
        
        # 상관관계 분석
        correlations = risk_manager.analyze_correlations(returns)
        logger.info("상관관계 분석 결과:")
        logger.info(correlations.tail())
        
        # 스트레스 테스트 시나리오
        scenarios = {
            'market_crash': {
                'BTC': -0.3,
                'ETH': -0.4
            },
            'volatility_spike': {
                'BTC': -0.15,
                'ETH': -0.2
            },
            'recovery': {
                'BTC': 0.2,
                'ETH': 0.25
            }
        }
        
        # 스트레스 테스트 수행
        stress_test_results = risk_manager.perform_stress_test(scenarios)
        logger.info("스트레스 테스트 결과:")
        for scenario, result in stress_test_results.items():
            logger.info(f"{scenario}: 최종 자본금 {result['final_capital']:.2f}, 수익률 {result['return']:.2%}")
        
        # 포지션 정보
        positions = {
            'BTC': {'size': 0.1, 'price': 50000.0},
            'ETH': {'size': 0.15, 'price': 3000.0}
        }
        
        # 포트폴리오 가치 계산
        portfolio_value = sum(pos['size'] * pos['price'] for pos in positions.values())
        
        # 리스크 모니터링
        risk_metrics = risk_manager.monitor_risk(portfolio_value, positions)
        logger.info("리스크 메트릭스:")
        logger.info(f"포트폴리오 가치: {risk_metrics['portfolio_value']:.2f}")
        logger.info(f"최대 손실폭: {risk_metrics['drawdown']:.2%}")
        
        if risk_metrics['warnings']:
            logger.warning("리스크 경고:")
            for warning in risk_metrics['warnings']:
                logger.warning(warning)
        
        # 포지션 사이즈 계산 예제
        for symbol in ['BTC', 'ETH']:
            volatility = returns[symbol].std()
            correlation = correlations[symbol].mean()
            position_size = risk_manager.calculate_position_size(
                price=positions[symbol]['price'],
                volatility=volatility,
                correlation=correlation
            )
            logger.info(f"{symbol} 권장 포지션 사이즈: {position_size:.2f}")
        
    except Exception as e:
        logger.error(f"리스크 관리 예제 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 