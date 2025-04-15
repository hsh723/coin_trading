import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.cm as cm

class PortfolioVisualizer:
    """포트폴리오 시각화 클래스
    
    다양한 포트폴리오 분석 및 시각화 기능 제공:
    - 자산 배분 시각화 (파이 차트, 바 차트)
    - 포트폴리오 성과 시각화
    - 효율적 프론티어 시각화
    - 다중 포트폴리오 비교
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Args:
            figsize: 기본 그림 크기
        """
        self.figsize = figsize
        # 시각화에 사용할 색상 팔레트 설정
        self.colors = sns.color_palette("Set3", 20)
        
    def plot_asset_allocation(self, weights: Dict[str, float], 
                              title: str = '포트폴리오 자산 배분',
                              plot_type: str = 'pie',
                              threshold: float = 0.03,
                              sort_by_weight: bool = True) -> None:
        """포트폴리오 자산 배분 시각화
        
        Args:
            weights: 자산 가중치 딕셔너리
            title: 그래프 제목
            plot_type: 그래프 유형 ('pie' 또는 'bar')
            threshold: 표시할 최소 가중치 임계값
            sort_by_weight: 가중치 기준 정렬 여부
        """
        # 임계값 이하 가중치 필터링 및 'Others'로 그룹화
        filtered_weights = {}
        others = 0.0
        
        for asset, weight in weights.items():
            if weight >= threshold:
                filtered_weights[asset] = weight
            else:
                others += weight
        
        # 'Others' 카테고리 추가 (있는 경우)
        if others > 0:
            filtered_weights['Others'] = others
        
        # 가중치 기준 정렬 (내림차순)
        if sort_by_weight:
            filtered_weights = dict(sorted(filtered_weights.items(), 
                                         key=lambda x: x[1], reverse=True))
        
        assets = list(filtered_weights.keys())
        values = list(filtered_weights.values())
        
        plt.figure(figsize=self.figsize)
        
        if plot_type == 'pie':
            # 파이 차트
            plt.pie(values, labels=assets, autopct='%1.1f%%', startangle=90,
                   colors=self.colors[:len(filtered_weights)])
            plt.axis('equal')
        elif plot_type == 'bar':
            # 바 차트
            positions = range(len(filtered_weights))
            plt.bar(positions, values, color=self.colors[:len(filtered_weights)])
            plt.xticks(positions, assets, rotation=45, ha='right')
            plt.ylabel('가중치')
            
            # 가중치 값 표시
            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f'{v:.1%}', ha='center')
        else:
            raise ValueError(f"지원하지 않는 그래프 유형: {plot_type}")
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_portfolio_performance(self, returns: pd.DataFrame, 
                                  weights: Dict[str, float],
                                  benchmark_returns: Optional[pd.Series] = None,
                                  benchmark_name: str = 'Benchmark',
                                  title: str = '포트폴리오 성과',
                                  show_drawdown: bool = True) -> None:
        """포트폴리오 성과 시각화
        
        Args:
            returns: 자산 수익률 데이터프레임
            weights: 자산 가중치 딕셔너리
            benchmark_returns: 벤치마크 수익률 시리즈 (선택적)
            benchmark_name: 벤치마크 이름
            title: 그래프 제목
            show_drawdown: 낙폭 그래프 표시 여부
        """
        # 가중치에 해당하는 자산만 선택
        portfolio_assets = list(weights.keys())
        filtered_returns = returns[portfolio_assets]
        
        # 포트폴리오 일별 수익률 계산
        weights_array = np.array([weights[asset] for asset in portfolio_assets])
        portfolio_returns = (filtered_returns * weights_array).sum(axis=1)
        
        # 누적 수익률 계산
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        # 포트폴리오 누적 수익률 그래프
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                label='Portfolio', linewidth=2)
        
        # 벤치마크 추가 (있는 경우)
        if benchmark_returns is not None:
            # 벤치마크 누적 수익률
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            ax1.plot(benchmark_cum_returns.index, benchmark_cum_returns.values, 
                    label=benchmark_name, linewidth=2, linestyle='--', alpha=0.7)
        
        ax1.set_title(title)
        ax1.set_ylabel('누적 수익률')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 낙폭 그래프
        if show_drawdown:
            # 낙폭 계산
            portfolio_drawdown = self._calculate_drawdown(portfolio_returns)
            ax2.fill_between(portfolio_drawdown.index, 0, portfolio_drawdown.values, 
                            color='red', alpha=0.3)
            ax2.set_ylabel('낙폭')
            ax2.set_xlabel('')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """낙폭 계산
        
        Args:
            returns: 수익률 시리즈
            
        Returns:
            낙폭 시리즈
        """
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        return drawdown
    
    def plot_efficient_frontier(self, risk_return_data: pd.DataFrame,
                              optimal_portfolio: Optional[Dict[str, float]] = None,
                              portfolios: Optional[List[Dict[str, Any]]] = None,
                              show_assets: bool = True,
                              title: str = '효율적 프론티어') -> None:
        """효율적 프론티어 시각화
        
        Args:
            risk_return_data: 효율적 프론티어 데이터 (Risk와 Return 열 포함)
            optimal_portfolio: 최적 포트폴리오 정보 (Risk, Return, Sharpe)
            portfolios: 표시할 추가 포트폴리오 목록
            show_assets: 개별 자산 표시 여부
            title: 그래프 제목
        """
        plt.figure(figsize=self.figsize)
        
        # 효율적 프론티어 플롯
        plt.plot(risk_return_data['Risk'], risk_return_data['Return'], 
                'b-', linewidth=2, label='효율적 프론티어')
        
        # 최적 포트폴리오 강조
        if optimal_portfolio is not None:
            plt.scatter(optimal_portfolio['Risk'], optimal_portfolio['Return'], 
                       s=100, c='red', marker='*', 
                       label=f'최적 포트폴리오 (Sharpe: {optimal_portfolio["Sharpe"]:.2f})')
        
        # 추가 포트폴리오 표시
        if portfolios is not None:
            for i, p in enumerate(portfolios):
                plt.scatter(p['Risk'], p['Return'], s=80, alpha=0.7,
                           label=f"{p['Name']} (Sharpe: {p['Sharpe']:.2f})")
        
        # 개별 자산 표시
        if show_assets and 'Assets' in risk_return_data.columns:
            assets = risk_return_data['Assets'].iloc[0]
            for asset, stats in assets.items():
                plt.scatter(stats['Risk'], stats['Return'], s=50, 
                           alpha=0.7, marker='o')
                plt.annotate(asset, (stats['Risk'], stats['Return']),
                            xytext=(5, 5), textcoords='offset points')
        
        plt.title(title)
        plt.xlabel('위험 (연간 표준편차)')
        plt.ylabel('기대 수익률 (연간)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, returns: pd.DataFrame, 
                               title: str = '자산 상관관계 행렬') -> None:
        """자산 상관관계 행렬 시각화
        
        Args:
            returns: 자산 수익률 데이터프레임
            title: 그래프 제목
        """
        correlation = returns.corr()
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   linewidths=0.5, fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, portfolios: List[Dict[str, float]], 
                               metrics: List[str] = None,
                               title: str = '포트폴리오 성과 지표 비교') -> None:
        """다중 포트폴리오 성과 지표 비교
        
        Args:
            portfolios: 포트폴리오 목록 (이름과 성과 지표 포함)
            metrics: 표시할 지표 목록 (None이면 모든 공통 지표)
            title: 그래프 제목
        """
        if not portfolios:
            raise ValueError("포트폴리오 목록이 비어 있습니다")
        
        # 기본 지표 설정
        if metrics is None:
            # 첫 번째 포트폴리오의 지표에서 '이름'을 제외한 모든 키 사용
            all_keys = set(portfolios[0].keys()) - {'Name'}
            # 모든 포트폴리오에 공통인 지표만 선택
            metrics = []
            for key in all_keys:
                if all(key in p for p in portfolios):
                    metrics.append(key)
        
        # 포트폴리오 이름 추출
        portfolio_names = [p.get('Name', f'Portfolio {i+1}') 
                          for i, p in enumerate(portfolios)]
        
        # 데이터 준비
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric] = [p.get(metric, 0) for p in portfolios]
        
        # 그래프 생성
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.figsize, 
                               sharex=True)
        
        if len(metrics) == 1:
            axes = [axes]  # 단일 지표인 경우 리스트로 변환
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            positions = range(len(portfolio_names))
            ax.bar(positions, metrics_data[metric], 
                  color=self.colors[:len(portfolio_names)])
            ax.set_ylabel(metric)
            
            # 값 표시
            for j, v in enumerate(metrics_data[metric]):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center')
            
            # 마지막 서브플롯에만 x축 레이블 표시
            if i == len(metrics) - 1:
                ax.set_xticks(positions)
                ax.set_xticklabels(portfolio_names, rotation=45, ha='right')
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_weights_comparison(self, portfolio_weights: List[Dict[str, Dict[str, float]]],
                              title: str = '포트폴리오 가중치 비교') -> None:
        """다중 포트폴리오 가중치 비교
        
        Args:
            portfolio_weights: 포트폴리오 가중치 목록
            title: 그래프 제목
        """
        if not portfolio_weights:
            raise ValueError("포트폴리오 가중치 목록이 비어 있습니다")
        
        # 포트폴리오 이름 및 자산 추출
        portfolio_names = [pw.get('Name', f'Portfolio {i+1}') 
                         for i, pw in enumerate(portfolio_weights)]
        
        # 모든 포트폴리오의 모든 자산 목록 생성
        all_assets = set()
        for pw in portfolio_weights:
            weights = pw.get('Weights', {})
            all_assets.update(weights.keys())
        
        all_assets = sorted(list(all_assets))
        
        # 데이터 준비
        weights_data = []
        for pw in portfolio_weights:
            weights = pw.get('Weights', {})
            weights_data.append([weights.get(asset, 0) for asset in all_assets])
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 누적 바 차트
        bottom = np.zeros(len(portfolio_names))
        for i, asset in enumerate(all_assets):
            values = [pw.get('Weights', {}).get(asset, 0) for pw in portfolio_weights]
            ax.bar(portfolio_names, values, label=asset, bottom=bottom, 
                  color=self.colors[i % len(self.colors)])
            bottom += values
        
        ax.set_title(title)
        ax.set_ylabel('가중치')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=min(5, len(all_assets)), frameon=True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_risk_contribution(self, weights: Dict[str, float], 
                              returns: pd.DataFrame,
                              title: str = '리스크 기여도 분석') -> None:
        """각 자산의 포트폴리오 리스크 기여도 시각화
        
        Args:
            weights: 자산 가중치 딕셔너리
            returns: 자산 수익률 데이터프레임
            title: 그래프 제목
        """
        # 공분산 행렬
        cov_matrix = returns.cov().values * 252
        
        # 자산 목록 및 가중치 벡터
        assets = list(weights.keys())
        weights_array = np.array([weights[asset] for asset in assets])
        
        # 포트폴리오 변동성
        portfolio_var = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # 한계 기여도 (MC)
        mc = np.dot(cov_matrix, weights_array) / portfolio_vol
        
        # 리스크 기여도 (RC)
        rc = weights_array * mc
        
        # 백분율 기여도
        pct_rc = rc / portfolio_vol
        
        # 결과 데이터 생성
        risk_data = pd.DataFrame({
            'Asset': assets,
            'Weight': weights_array,
            'Marginal_Contribution': mc,
            'Risk_Contribution': rc,
            'Percent_Contribution': pct_rc * 100
        })
        
        # 기여도 기준 정렬
        risk_data = risk_data.sort_values('Percent_Contribution', ascending=False)
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # 가중치 vs 리스크 기여도
        positions = range(len(assets))
        width = 0.35
        
        ax1.bar([p - width/2 for p in positions], risk_data['Weight'], 
               width=width, label='가중치', alpha=0.7)
        ax1.bar([p + width/2 for p in positions], risk_data['Percent_Contribution']/100, 
               width=width, label='리스크 기여도 %', alpha=0.7)
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels(risk_data['Asset'], rotation=45, ha='right')
        ax1.set_ylabel('비율')
        ax1.set_title('가중치 vs 리스크 기여도')
        ax1.legend()
        
        # 파이 차트: 리스크 기여도
        ax2.pie(risk_data['Percent_Contribution'], labels=risk_data['Asset'], 
               autopct='%1.1f%%', startangle=90,
               colors=self.colors[:len(assets)])
        ax2.axis('equal')
        ax2.set_title('리스크 기여도 분포')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return risk_data 