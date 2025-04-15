import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import List, Dict, Union, Tuple, Optional, Any, Callable
import logging
from matplotlib.figure import Figure
import os
import pymc3 as pm
from scipy import stats
from functools import partial

# 로깅 설정
logger = logging.getLogger(__name__)

class BayesianModelInterpreter:
    """
    베이지안 시계열 모델 해석 클래스
    
    다양한 해석 및 분석 기능을 제공합니다:
    - 모델 파라미터 통계 분석
    - 변수 중요도 계산
    - 사후 예측 검사
    - 민감도 분석
    - 모델 불확실성 해석
    """
    
    def __init__(self, 
                 save_dir: str = "./interpretations",
                 fig_size: Tuple[int, int] = (12, 8)):
        """
        베이지안 모델 해석기 초기화
        
        Args:
            save_dir: 해석 결과 저장 디렉토리
            fig_size: 기본 그래프 크기
        """
        self.save_dir = save_dir
        self.fig_size = fig_size
        
        # 저장 디렉토리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"해석 결과 저장 디렉토리 생성: {save_dir}")
    
    def summarize_parameters(self, 
                           trace: Any, 
                           var_names: Optional[List[str]] = None,
                           credible_interval: float = 0.94) -> pd.DataFrame:
        """
        모델 파라미터의 사후 분포 요약 통계 계산
        
        Args:
            trace: PyMC3 트레이스 객체
            var_names: 요약할 변수 이름 목록 (None이면 전체 변수)
            credible_interval: 신뢰 구간 (0~1 사이)
            
        Returns:
            요약 통계 데이터프레임
        """
        # arviz를 사용한 사후 분포 요약
        summary = az.summary(
            trace, 
            var_names=var_names, 
            credible_interval=credible_interval,
            round_to=4
        )
        
        # 결과 반환
        logger.info(f"모델 파라미터 요약 통계 생성 완료: {len(summary)} 파라미터")
        return summary
    
    def plot_parameter_importance(self, 
                                trace: Any,
                                var_names: Optional[List[str]] = None,
                                target_var: str = None,
                                title: str = "파라미터 중요도",
                                save_path: Optional[str] = None) -> Figure:
        """
        변수 중요도 시각화 (표준화된 평균 절대값 기준)
        
        Args:
            trace: PyMC3 트레이스 객체
            var_names: 중요도를 계산할 변수 이름 목록
            target_var: 중요도 기준이 되는 타겟 변수 (예: 'y_pred')
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            그래프 객체
        """
        # 변수 중요도 계산
        summary = self.summarize_parameters(trace, var_names)
        
        # 평균 절대값 기준으로 중요도 계산
        importance = summary['mean'].abs()
        
        # 중요도 기준 내림차순 정렬
        importance = importance.sort_values(ascending=False)
        
        # 시각화
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # 바 그래프
        bars = ax.barh(importance.index, importance.values, color='skyblue')
        
        # 수치 표시
        for i, (name, value) in enumerate(zip(importance.index, importance.values)):
            ax.text(value + 0.01, i, f"{value:.4f}", va='center')
        
        # 제목 및 레이블
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('평균 절대값 중요도', fontsize=12)
        ax.set_ylabel('파라미터', fontsize=12)
        
        # 그리드 및 테마
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"파라미터 중요도 그래프 저장 완료: {full_path}")
        
        return fig
    
    def posterior_predictive_check(self,
                                 model: Any,
                                 trace: Any,
                                 observed_data: Union[pd.Series, np.ndarray],
                                 n_samples: int = 100,
                                 title: str = "사후 예측 검사",
                                 save_path: Optional[str] = None) -> Figure:
        """
        사후 예측 검사 수행 및 시각화
        
        Args:
            model: PyMC3 모델 객체
            trace: PyMC3 트레이스 객체
            observed_data: 관측된 실제 데이터
            n_samples: 사후 예측에서 생성할 샘플 수
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            그래프 객체
        """
        # 사후 예측 검사 수행
        with model:
            # 사후 예측 샘플 생성
            ppc = pm.sample_posterior_predictive(trace, samples=n_samples)
        
        # 사후 예측 결과 시각화
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # 사후 예측 데이터
        if 'y' in ppc:
            posterior_samples = ppc['y']
        elif 'y_pred' in ppc:
            posterior_samples = ppc['y_pred']
        else:
            # 첫 번째 변수 사용
            var_name = list(ppc.keys())[0]
            posterior_samples = ppc[var_name]
            logger.warning(f"y 또는 y_pred 변수를 찾을 수 없어 {var_name}을(를) 사용합니다.")
        
        # 관측 데이터 형식 변환
        if isinstance(observed_data, pd.Series):
            observed = observed_data.values
        else:
            observed = observed_data
        
        # 사후 예측 분포 시각화
        for i in range(min(20, n_samples)):  # 너무 많은 라인은 시각적으로 혼잡함
            ax.plot(posterior_samples[i], color='blue', alpha=0.1)
        
        # 예측 구간 계산 및 표시
        pred_mean = np.mean(posterior_samples, axis=0)
        pred_std = np.std(posterior_samples, axis=0)
        
        # 평균과 90% 신뢰 구간 표시
        ax.plot(pred_mean, color='blue', linewidth=2, label='예측 평균')
        ax.fill_between(
            np.arange(len(pred_mean)),
            pred_mean - 1.645 * pred_std,
            pred_mean + 1.645 * pred_std,
            color='blue', alpha=0.2, label='90% 신뢰 구간'
        )
        
        # 실제 관측 데이터 표시
        ax.plot(observed, color='red', linewidth=2, label='관측 데이터')
        
        # 제목 및 레이블
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('시간 인덱스', fontsize=12)
        ax.set_ylabel('값', fontsize=12)
        
        # 범례 및 그리드
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"사후 예측 검사 그래프 저장 완료: {full_path}")
        
        return fig
    
    def analyze_uncertainty(self,
                          forecast: np.ndarray,
                          lower: np.ndarray,
                          upper: np.ndarray,
                          times: Optional[Union[pd.DatetimeIndex, List, np.ndarray]] = None,
                          title: str = "예측 불확실성 분석",
                          save_path: Optional[str] = None) -> Tuple[Figure, Dict[str, float]]:
        """
        예측 불확실성 분석 및 시각화
        
        Args:
            forecast: 예측값
            lower: 하한 신뢰구간
            upper: 상한 신뢰구간
            times: 예측 시점 (인덱스 또는 날짜)
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            그래프 객체와 불확실성 지표 딕셔너리
        """
        # 예측 불확실성 계산
        uncertainty_width = upper - lower
        uncertainty_ratio = uncertainty_width / forecast
        
        # 불확실성 지표 계산
        uncertainty_metrics = {
            'mean_width': np.mean(uncertainty_width),
            'max_width': np.max(uncertainty_width),
            'min_width': np.min(uncertainty_width),
            'mean_ratio': np.mean(uncertainty_ratio),
            'max_ratio': np.max(uncertainty_ratio),
            'min_ratio': np.min(uncertainty_ratio),
            'increasing_uncertainty': np.corrcoef(np.arange(len(uncertainty_width)), uncertainty_width)[0, 1]
        }
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size, sharex=True)
        
        # x축 데이터 설정
        if times is None:
            x = np.arange(len(forecast))
        else:
            x = times
        
        # 상단 그래프: 예측 및 신뢰 구간
        ax1.plot(x, forecast, color='blue', linewidth=2, label='예측값')
        ax1.fill_between(x, lower, upper, color='blue', alpha=0.2, label='신뢰 구간')
        
        # 제목 및 레이블
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('예측값', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 하단 그래프: 불확실성 너비
        ax2.plot(x, uncertainty_width, color='red', linewidth=2, label='불확실성 너비')
        # 불확실성 추세선
        z = np.polyfit(np.arange(len(uncertainty_width)), uncertainty_width, 1)
        p = np.poly1d(z)
        ax2.plot(x, p(np.arange(len(uncertainty_width))), 
                linestyle='--', color='darkred', label=f'추세 (상관계수: {uncertainty_metrics["increasing_uncertainty"]:.3f})')
        
        # 레이블 및 범례
        ax2.set_xlabel('시간', fontsize=12)
        ax2.set_ylabel('불확실성 너비', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"불확실성 분석 그래프 저장 완료: {full_path}")
        
        return fig, uncertainty_metrics
    
    def parameter_sensitivity(self,
                            model_func: Callable,
                            baseline_params: Dict[str, Any],
                            param_ranges: Dict[str, List[float]],
                            metric_func: Optional[Callable] = None,
                            data: Optional[Union[pd.Series, np.ndarray]] = None,
                            title: str = "파라미터 민감도 분석",
                            save_path: Optional[str] = None) -> Figure:
        """
        모델 파라미터 민감도 분석 및 시각화
        
        Args:
            model_func: 모델 생성 함수(파라미터를 받아 모델 반환)
            baseline_params: 기본 파라미터 딕셔너리
            param_ranges: 분석할 파라미터별 범위 값
            metric_func: 성능 지표 계산 함수 (예측과 실제 데이터 기반)
            data: 평가에 사용할 데이터
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            그래프 객체
        """
        # 결과를 저장할 딕셔너리
        sensitivity_results = {}
        
        # 각 파라미터별 민감도 분석
        for param_name, param_values in param_ranges.items():
            results = []
            
            # 기준 예측값 저장
            baseline_model = model_func(**baseline_params)
            baseline_forecast = None
            
            # 각 파라미터 값에 대한 민감도 분석
            for value in param_values:
                # 현재 파라미터 값으로 모델 설정
                test_params = baseline_params.copy()
                test_params[param_name] = value
                
                # 모델 생성
                model = model_func(**test_params)
                
                # 예측 수행
                forecast = None
                if hasattr(model, "predict"):
                    try:
                        forecast, _, _ = model.predict(n_forecast=len(data) if data is not None else 10)
                    except Exception as e:
                        logger.error(f"예측 실패: {str(e)}")
                        forecast = None
                
                if baseline_forecast is None and forecast is not None:
                    baseline_forecast = forecast
                
                # 지표 계산 (제공된 함수가 있는 경우)
                if metric_func is not None and data is not None and forecast is not None:
                    try:
                        metric = metric_func(data, forecast)
                    except Exception as e:
                        logger.error(f"지표 계산 실패: {str(e)}")
                        metric = np.nan
                else:
                    # 기준 예측과의 차이로 민감도 계산
                    if forecast is not None and baseline_forecast is not None:
                        metric = np.mean(np.abs(forecast - baseline_forecast))
                    else:
                        metric = np.nan
                
                results.append((value, metric))
            
            # 결과 저장
            sensitivity_results[param_name] = results
        
        # 시각화
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(n_params, 1, figsize=(self.fig_size[0], n_params * 4), sharex=False)
        
        # 단일 파라미터인 경우 axes를 리스트로 변환
        if n_params == 1:
            axes = [axes]
        
        # 각 파라미터별 민감도 그래프
        for i, (param_name, results) in enumerate(sensitivity_results.items()):
            ax = axes[i]
            
            # 결과 데이터 분리
            param_values = [r[0] for r in results]
            metric_values = [r[1] for r in results]
            
            # 파라미터별 민감도 그래프
            ax.plot(param_values, metric_values, marker='o', linestyle='-', color='purple')
            
            # 기준 파라미터 값 표시
            baseline_idx = None
            try:
                baseline_value = baseline_params[param_name]
                if baseline_value in param_values:
                    baseline_idx = param_values.index(baseline_value)
                    ax.scatter([param_values[baseline_idx]], [metric_values[baseline_idx]], 
                             color='red', s=100, zorder=5, label='기준값')
            except (KeyError, ValueError):
                pass
            
            # 제목 및 레이블
            ax.set_title(f"파라미터 '{param_name}' 민감도", fontsize=12)
            ax.set_xlabel(f"{param_name} 값", fontsize=10)
            ax.set_ylabel("민감도 지표", fontsize=10)
            
            # 그리드 및 범례
            ax.grid(True, alpha=0.3)
            if baseline_idx is not None:
                ax.legend()
        
        # 전체 그래프 제목
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # rect는 suptitle 공간 확보
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"파라미터 민감도 그래프 저장 완료: {full_path}")
        
        return fig
    
    def analyze_forecast_components(self,
                                  model: Any,
                                  n_forecast: int = 10,
                                  component_names: Optional[List[str]] = None,
                                  dates: Optional[Union[pd.DatetimeIndex, List]] = None,
                                  title: str = "예측 구성요소 분석",
                                  save_path: Optional[str] = None) -> Figure:
        """
        구조적 시계열 모델의 예측 구성요소 분석 및 시각화
        
        Args:
            model: 구조적 시계열 모델 객체 (trend, seasonality 등의 구성요소가 있어야 함)
            n_forecast: 예측 기간
            component_names: 구성요소 이름 목록
            dates: 예측 날짜
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            그래프 객체
        """
        # 기본 구성요소 이름
        if component_names is None:
            component_names = ['trend', 'seasonality', 'regression', 'autoregressive', 'residual']
        
        # 구성요소 데이터 추출
        components = {}
        
        # 모델에서 구성요소 데이터 추출 시도
        for name in component_names:
            if hasattr(model, f"get_{name}_component"):
                method = getattr(model, f"get_{name}_component")
                try:
                    component = method(n_forecast=n_forecast)
                    if component is not None:
                        components[name] = component
                except Exception as e:
                    logger.warning(f"'{name}' 구성요소 추출 실패: {str(e)}")
            elif hasattr(model, name):
                # 속성으로 존재하는 경우
                attr = getattr(model, name)
                if callable(attr):
                    try:
                        component = attr(n_forecast=n_forecast)
                        components[name] = component
                    except Exception:
                        pass
                elif isinstance(attr, (np.ndarray, list)):
                    components[name] = attr
        
        # 구성요소가 없는 경우
        if not components:
            logger.warning("분석할 구성요소를 찾을 수 없습니다.")
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, "구성요소를 찾을 수 없습니다.", ha='center', va='center', fontsize=14)
            return fig
        
        # 시각화
        n_components = len(components)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=(self.fig_size[0], (n_components + 1) * 3), sharex=True)
        
        # x축 데이터 설정
        if dates is None:
            x = np.arange(n_forecast)
        else:
            x = dates
        
        # 총 예측값 계산 (모든 구성요소의 합)
        total_forecast = np.zeros(n_forecast)
        for name, component in components.items():
            if len(component) >= n_forecast:
                total_forecast += component[:n_forecast]
        
        # 첫 번째 그래프: 총 예측값
        axes[0].plot(x, total_forecast, color='black', linewidth=2, label='총 예측')
        axes[0].set_title("총 예측값", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 각 구성요소 그래프
        colors = plt.cm.tab10.colors
        for i, (name, component) in enumerate(components.items()):
            ax = axes[i + 1]
            
            # 구성요소 플롯
            color = colors[i % len(colors)]
            if len(component) >= n_forecast:
                ax.plot(x, component[:n_forecast], color=color, linewidth=2, label=name)
            
            # 제목 및 레이블
            ax.set_title(f"{name} 구성요소", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 전체 그래프 제목 및 레이아웃
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # rect는 suptitle 공간 확보
        
        # 그래프 저장
        if save_path is not None:
            save_file = save_path if save_path.endswith(('.png', '.jpg', '.svg', '.pdf')) else f"{save_path}.png"
            full_path = os.path.join(self.save_dir, save_file)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"예측 구성요소 그래프 저장 완료: {full_path}")
        
        return fig
    
    def generate_interpretation_report(self,
                                     model: Any,
                                     trace: Any,
                                     observed_data: Union[pd.Series, np.ndarray],
                                     forecast: np.ndarray,
                                     lower: np.ndarray,
                                     upper: np.ndarray,
                                     report_path: str = "model_interpretation_report.html") -> str:
        """
        종합적인 모델 해석 보고서 생성
        
        Args:
            model: 모델 객체
            trace: PyMC3 트레이스 객체
            observed_data: 관측 데이터
            forecast: 예측값
            lower: 하한 신뢰구간
            upper: 상한 신뢰구간
            report_path: 보고서 저장 경로
            
        Returns:
            생성된 보고서 경로
        """
        # 보고서 제목 및 기본 정보
        model_type = type(model).__name__
        params_summary = self.summarize_parameters(trace)
        
        # 불확실성 분석
        _, uncertainty_metrics = self.analyze_uncertainty(forecast, lower, upper)
        
        # HTML 보고서 생성
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>베이지안 모델 해석 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ font-weight: bold; margin-right: 10px; }}
                .value {{ color: #3498db; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>베이지안 모델 해석 보고서</h1>
            
            <div class="section">
                <h2>모델 정보</h2>
                <p><span class="metric">모델 유형:</span> <span class="value">{model_type}</span></p>
                <p><span class="metric">데이터 크기:</span> <span class="value">{len(observed_data)}</span></p>
                <p><span class="metric">예측 기간:</span> <span class="value">{len(forecast)}</span></p>
            </div>
            
            <div class="section">
                <h2>모델 파라미터 요약</h2>
                <table>
                    <tr>
                        <th>파라미터</th>
                        <th>평균</th>
                        <th>표준편차</th>
                        <th>신뢰구간</th>
                    </tr>
        """
        
        # 파라미터 테이블 생성
        for param, row in params_summary.iterrows():
            # 신뢰구간 컬럼명 찾기
            hdi_cols = [col for col in params_summary.columns if 'hdi' in col.lower()]
            hdi_str = ""
            if hdi_cols:
                lower_hdi = row[hdi_cols[0]]
                upper_hdi = row[hdi_cols[1]]
                hdi_str = f"[{lower_hdi}, {upper_hdi}]"
            
            report_content += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{row['mean']}</td>
                        <td>{row['sd']}</td>
                        <td>{hdi_str}</td>
                    </tr>
            """
        
        report_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>불확실성 분석</h2>
                <table>
                    <tr>
                        <th>지표</th>
                        <th>값</th>
                    </tr>
        """
        
        # 불확실성 지표 테이블 생성
        for metric, value in uncertainty_metrics.items():
            report_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """
        
        report_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>해석 및 결론</h2>
                <p>
                    이 보고서는 베이지안 시계열 모델의 해석 결과를 제공합니다. 
                    모델 파라미터의 사후 분포와 불확실성 분석을 통해 예측 결과의 
                    신뢰도와 특성을 확인할 수 있습니다.
                </p>
                <p>
                    불확실성 구간의 증가/감소 추세를 통해 미래 예측의 신뢰도 변화를 
                    분석할 수 있으며, 주요 파라미터의 영향을 파악할 수 있습니다.
                </p>
            </div>
        </body>
        </html>
        """
        
        # 보고서 저장
        full_path = os.path.join(self.save_dir, report_path)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"모델 해석 보고서 생성 완료: {full_path}")
        return full_path 