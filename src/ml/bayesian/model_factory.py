from typing import Dict, Any, Optional, Union, Type
from .bayesian_time_series_predictor import BayesianTimeSeriesPredictor
from .gaussian_process_model import GaussianProcessModel
from .structural_time_series import StructuralTimeSeriesModel
import logging

logger = logging.getLogger(__name__)

class BayesianModelFactory:
    """
    베이지안 시계열 예측 모델 팩토리 클래스
    
    다양한 베이지안 시계열 모델을 생성하는 팩토리 패턴 구현
    """
    
    # 등록된 모델 클래스 저장
    _models: Dict[str, Type] = {
        "ar": BayesianTimeSeriesPredictor,
        "gp": GaussianProcessModel,
        "structural": StructuralTimeSeriesModel
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type) -> None:
        """
        새로운 모델 클래스 등록
        
        Args:
            name: 모델 이름
            model_class: 모델 클래스
        """
        if name in cls._models:
            logger.warning(f"모델 '{name}'이(가) 이미 등록되어 있어 덮어씁니다.")
        
        cls._models[name] = model_class
        logger.info(f"새 모델 '{name}'이(가) 등록되었습니다.")
    
    @classmethod
    def get_model(cls, model_type: str, **kwargs) -> Union[BayesianTimeSeriesPredictor, 
                                                        GaussianProcessModel, 
                                                        StructuralTimeSeriesModel]:
        """
        모델 인스턴스 생성
        
        Args:
            model_type: 모델 유형 ('ar', 'gp', 'structural')
            **kwargs: 모델별 추가 파라미터
            
        Returns:
            모델 인스턴스
        
        Raises:
            ValueError: 지원하지 않는 모델 유형인 경우
        """
        if model_type not in cls._models:
            available_models = ", ".join(cls._models.keys())
            raise ValueError(f"모델 유형 '{model_type}'은(는) 지원되지 않습니다. 사용 가능한 모델: {available_models}")
        
        # 모델 인스턴스 생성
        model_class = cls._models[model_type]
        
        # 모델별 특수 파라미터 처리
        if model_type == "ar":
            default_params = {
                "model_type": "ar",
                "seasonality": False,
                "num_seasons": 7,
                "ar_order": 1
            }
        elif model_type == "gp":
            default_params = {
                "kernel_type": "rbf",
                "seasonality": False,
                "period": 7,
                "trend": True
            }
        elif model_type == "structural":
            default_params = {
                "level": True,
                "trend": True,
                "seasonality": False,
                "season_period": 7,
                "damped_trend": False
            }
        else:
            default_params = {}
        
        # 기본 파라미터에 사용자 파라미터 병합
        for key, value in kwargs.items():
            default_params[key] = value
        
        # 모델 생성
        try:
            model = model_class(**default_params)
            logger.info(f"'{model_type}' 모델 인스턴스가 생성되었습니다.")
            return model
        except Exception as e:
            logger.error(f"모델 생성 중 오류 발생: {str(e)}")
            raise 