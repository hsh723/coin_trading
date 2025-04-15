#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
포트폴리오 최적화 시스템 예제 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import yfinance as yf

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.portfolio_visualizer import PortfolioVisualizer
from src.portfolio.dynamic_allocation_manager import DynamicAllocationManager
from src.utils.logger import setup_logger

# 로거 설정
logger = setup_logger('portfolio_optimizer_example')

def get_stock_data(tickers, start_date=None, end_date=None):
    """
    주식 가격 데이터 수집
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365 * 5)  # 5년 데이터
    if end_date is None:
        end_date = datetime.now()
        
    logger.info(f"주식 데이터 다운로드 중: {tickers}")
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Adj Close 열만 선택하고 열 이름 정리
    prices = data['Adj Close']
    
    logger.info(f"다운로드 완료: {len(prices)} 행, {len(prices.columns)} 열")
    return prices

def calculate_returns(prices, log_returns=False):
    """
    수익률 계산
    """
    if log_returns:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    returns.dropna(inplace=True)
    return returns

def run_optimization_example():
    """
    포트폴리오 최적화 예제 실행
    """
    logger.info("포트폴리오 최적화 예제 시작")
    
    # 1. 데이터 수집: S&P 500 주요 종목 및 ETF 선택
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 
               'DIS', 'HD', 'MRK', 'INTC', 'VZ', 'ADBE', 'CSCO', 'NFLX', 'PFE', 'KO']
    
    # 자산군 ETF
    etfs = ['SPY', 'QQQ', 'GLD', 'AGG', 'VNQ', 'VEA', 'VWO', 'LQD', 'TLT']
    
    # 데이터 다운로드
    stock_prices = get_stock_data(tickers + etfs, 
                                  start_date=datetime.now() - timedelta(days=365 * 3))  # 3년 데이터
    
    # 2. 수익률 계산
    returns = calculate_returns(stock_prices)
    
    # 3. 포트폴리오 최적화
    logger.info("포트폴리오 최적화 수행")
    
    # 시각화 객체 생성
    visualizer = PortfolioVisualizer()
    
    # 다양한 최적화 방법으로 포트폴리오 구성
    optimizer = PortfolioOptimizer(risk_free_rate=0.025)  # 현재 무위험 수익률 반영
    
    # 최적 샤프 비율 포트폴리오
    sharpe_weights = optimizer.optimize(returns, method='sharpe')
    
    logger.info("샤프 비율 최적화 포트폴리오 가중치:")
    for asset, weight in sorted(sharpe_weights.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:  # 1% 이상만 표시
            logger.info(f"  {asset}: {weight:.2%}")
    
    # 최소 분산 포트폴리오
    min_var_weights = optimizer.optimize(returns, method='min_var')
    
    logger.info("최소 분산 포트폴리오 가중치:")
    for asset, weight in sorted(min_var_weights.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:  # 1% 이상만 표시
            logger.info(f"  {asset}: {weight:.2%}")
    
    # 리스크 패리티 포트폴리오
    risk_parity_weights = optimizer.optimize(returns, method='risk_parity')
    
    logger.info("리스크 패리티 포트폴리오 가중치:")
    for asset, weight in sorted(risk_parity_weights.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:  # 1% 이상만 표시
            logger.info(f"  {asset}: {weight:.2%}")
    
    # 최대 다각화 포트폴리오
    max_div_weights = optimizer.optimize(returns, method='max_div')
    
    logger.info("최대 다각화 포트폴리오 가중치:")
    for asset, weight in sorted(max_div_weights.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:  # 1% 이상만 표시
            logger.info(f"  {asset}: {weight:.2%}")
    
    # 4. 효율적 프론티어 계산
    logger.info("효율적 프론티어 계산")
    efficient_frontier = optimizer.calculate_efficient_frontier(returns, n_points=50)
    
    # 5. 포트폴리오 시각화
    # 효율적 프론티어 시각화
    logger.info("효율적 프론티어 시각화")
    optimizer.plot_efficient_frontier(title='포트폴리오 효율적 프론티어')
    
    # 다양한 포트폴리오 구성 비교
    logger.info("포트폴리오 구성 비교")
    
    # 샤프 비율 최적화 포트폴리오 시각화
    visualizer.plot_asset_allocation(sharpe_weights, 
                                    title='샤프 비율 최적화 포트폴리오', 
                                    plot_type='pie',
                                    threshold=0.03)
    
    # 여러 포트폴리오 가중치 비교
    portfolios = [
        {'Name': '최적 샤프 비율', 'Weights': sharpe_weights},
        {'Name': '최소 분산', 'Weights': min_var_weights},
        {'Name': '리스크 패리티', 'Weights': risk_parity_weights},
        {'Name': '최대 다각화', 'Weights': max_div_weights}
    ]
    
    visualizer.plot_weights_comparison(portfolios, title='포트폴리오 비교')
    
    # 6. 포트폴리오 성과 분석
    logger.info("포트폴리오 성과 분석")
    
    # 각 포트폴리오의 성과 지표 계산
    portfolio_stats = []
    
    for p in portfolios:
        weights = p['Weights']
        stats = optimizer.get_portfolio_stats(weights, returns)
        stats['Name'] = p['Name']
        portfolio_stats.append(stats)
    
    # 성과 지표 비교 시각화
    metrics_to_show = ['return', 'risk', 'sharpe']
    visualizer.plot_performance_metrics(portfolio_stats, metrics=metrics_to_show,
                                     title='포트폴리오 성과 지표 비교')
    
    # 7. 리스크 기여도 분석 (샤프 비율 최적화 포트폴리오)
    logger.info("리스크 기여도 분석")
    visualizer.plot_risk_contribution(sharpe_weights, returns, 
                                    title='샤프 비율 최적화 포트폴리오 리스크 기여도')
    
    # 8. 동적 자산 배분 시뮬레이션
    logger.info("동적 자산 배분 시뮬레이션")
    
    # ETF만으로 포트폴리오 구성
    etf_returns = returns[etfs]
    etf_optimizer = PortfolioOptimizer(risk_free_rate=0.025)
    etf_weights = etf_optimizer.optimize(etf_returns, method='sharpe')
    
    # 동적 자산 배분 관리자 설정
    dynamic_config = {
        'rebalance_threshold': 0.05,  # 5% 임계값
        'rebalance_period': 30,       # 30일마다 재조정
        'min_rebalance_interval': 10, # 최소 10일 간격
        'trading_cost': 0.001         # 0.1% 거래 비용
    }
    
    allocation_manager = DynamicAllocationManager(etf_weights, config=dynamic_config)
    
    # 시뮬레이션 실행 (ETF 가격 데이터로)
    etf_prices = stock_prices[etfs]
    
    simulation_results = allocation_manager.simulate_rebalance_strategy(
        etf_prices, 
        initial_value=10000,  # $10,000 초기 투자
        start_date=etf_prices.index[0],
        end_date=etf_prices.index[-1]
    )
    
    # 시뮬레이션 결과 시각화
    logger.info("시뮬레이션 결과 시각화")
    
    # 포트폴리오 가치 추이
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_results.index, simulation_results['portfolio_value'], label='포트폴리오 가치')
    
    # 재조정 시점 표시
    rebalance_dates = simulation_results[simulation_results['rebalanced'] == 1].index
    for date in rebalance_dates:
        plt.axvline(x=date, color='r', linestyle='--', alpha=0.3)
    
    plt.title('동적 자산 배분 시뮬레이션 결과')
    plt.xlabel('날짜')
    plt.ylabel('포트폴리오 가치 ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 누적 수익률
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_results.index, simulation_results['cumulative_return'], 
            label='포트폴리오 누적 수익률')
    
    # 벤치마크(SPY) 수익률 추가
    spy_returns = etf_prices['SPY'].pct_change().dropna()
    spy_cum_returns = (1 + spy_returns).cumprod() - 1
    plt.plot(spy_cum_returns.index, spy_cum_returns, 
            label='SPY 누적 수익률', linestyle='--')
    
    plt.title('포트폴리오 vs 벤치마크 성과')
    plt.xlabel('날짜')
    plt.ylabel('누적 수익률')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 자산 가중치 변화
    weight_cols = [col for col in simulation_results.columns if col.startswith('weight_')]
    weights_df = simulation_results[weight_cols]
    
    # 열 이름 정리
    weights_df.columns = [col.replace('weight_', '') for col in weights_df.columns]
    
    plt.figure(figsize=(12, 6))
    weights_df.plot.area(figsize=(12, 6), alpha=0.7)
    
    # 재조정 시점 표시
    for date in rebalance_dates:
        plt.axvline(x=date, color='k', linestyle='--', alpha=0.3)
    
    plt.title('자산 가중치 변화')
    plt.xlabel('날짜')
    plt.ylabel('가중치')
    plt.grid(False)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    
    # 9. 결과 요약
    logger.info("시뮬레이션 결과 요약")
    
    # 연간 수익률, 변동성, 샤프 비율 계산
    daily_returns = simulation_results['return'].dropna()
    annual_return = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - 0.025) / annual_vol
    
    logger.info(f"연간 수익률: {annual_return:.2%}")
    logger.info(f"연간 변동성: {annual_vol:.2%}")
    logger.info(f"샤프 비율: {sharpe_ratio:.3f}")
    
    # 최대 낙폭 계산
    cumulative = (1 + daily_returns).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1)
    max_drawdown = drawdown.min()
    
    logger.info(f"최대 낙폭: {max_drawdown:.2%}")
    
    # 재조정 통계
    num_rebalances = len(rebalance_dates)
    total_cost = simulation_results['rebalance_cost'].sum()
    
    logger.info(f"총 재조정 횟수: {num_rebalances}")
    logger.info(f"총 거래 비용: ${total_cost:.2f}")

if __name__ == "__main__":
    run_optimization_example() 