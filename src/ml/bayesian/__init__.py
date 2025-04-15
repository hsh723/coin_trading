"""
베이지안 시계열 예측 패키지

암호화폐 시장 및 기타 시계열 데이터에 대한 베이지안 예측 모델을 제공합니다.
"""

from .bayesian_time_series_predictor import BayesianTimeSeriesPredictor
from .gaussian_process_model import GaussianProcessModel
from .structural_time_series import StructuralTimeSeriesModel
from .model_factory import BayesianModelFactory
from .ensemble_model import BayesianEnsembleModel
from .online_learner import OnlineBayesianLearner
from .model_visualization import BayesianModelVisualizer
from .model_interpretation import BayesianModelInterpreter
from .anomaly_detection import BayesianAnomalyDetector
from .model_deployment import ModelDeployer
from .backtesting import BayesianBacktester
from .strategy_optimizer import StrategyOptimizer

__version__ = '0.1.0'

# 편의를 위한 별칭
AR = BayesianTimeSeriesPredictor
GP = GaussianProcessModel
Structural = StructuralTimeSeriesModel
ModelFactory = BayesianModelFactory
Ensemble = BayesianEnsembleModel
OnlineLearner = OnlineBayesianLearner
Visualizer = BayesianModelVisualizer
Interpreter = BayesianModelInterpreter
AnomalyDetector = BayesianAnomalyDetector
Deployer = ModelDeployer
Backtester = BayesianBacktester
Optimizer = StrategyOptimizer 