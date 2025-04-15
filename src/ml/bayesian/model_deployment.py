import os
import json
import logging
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import threading
import queue
import fastapi
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram
import mlflow
import mlflow.pytorch
import torch
import joblib

from .model_factory import BayesianModelFactory
from .model_interpretation import BayesianModelInterpreter
from .anomaly_detection import BayesianAnomalyDetector

logger = logging.getLogger(__name__)

class ModelDeployer:
    """
    베이지안 시계열 예측 모델 배포 시스템
    
    주요 기능:
    - 모델 저장/로드
    - API 서빙
    - 버전 관리
    - 성능 모니터링
    - 실시간 예측
    - 모델 업데이트
    """
    
    def __init__(self,
                 model_dir: str = "./models",
                 api_host: str = "0.0.0.0",
                 api_port: int = 8000,
                 monitoring_interval: int = 60,
                 max_models: int = 5):
        """
        모델 배포 시스템 초기화
        
        Args:
            model_dir: 모델 저장 디렉토리
            api_host: API 서버 호스트
            api_port: API 서버 포트
            monitoring_interval: 모니터링 간격 (초)
            max_models: 최대 저장 모델 수
        """
        self.model_dir = model_dir
        self.api_host = api_host
        self.api_port = api_port
        self.monitoring_interval = monitoring_interval
        self.max_models = max_models
        
        # 모델 저장소 초기화
        os.makedirs(model_dir, exist_ok=True)
        
        # 모델 캐시
        self.model_cache = {}
        self.model_metadata = {}
        
        # 성능 메트릭스
        self._init_metrics()
        
        # 예측 큐
        self.prediction_queue = queue.Queue()
        self.prediction_thread = None
        
        # API 서버
        self.app = FastAPI(title="Bayesian Model API")
        self._setup_api_routes()
        
        # 모니터링 스레드
        self.monitoring_thread = None
        self.is_running = False
    
    def _init_metrics(self):
        """Prometheus 메트릭스 초기화"""
        # 예측 관련 메트릭스
        self.predictions_total = Counter('predictions_total', 'Total number of predictions')
        self.prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
        self.prediction_errors = Counter('prediction_errors_total', 'Total number of prediction errors')
        
        # 모델 관련 메트릭스
        self.model_versions = Gauge('model_versions', 'Number of model versions')
        self.model_memory_usage = Gauge('model_memory_usage_bytes', 'Model memory usage in bytes')
        self.model_performance = Gauge('model_performance_score', 'Model performance score')
    
    def _setup_api_routes(self):
        """FastAPI 라우트 설정"""
        
        class PredictionRequest(BaseModel):
            data: List[float]
            model_version: Optional[str] = None
        
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                start_time = time.time()
                result = self.predict(request.data, request.model_version)
                latency = time.time() - start_time
                self.prediction_latency.observe(latency)
                self.predictions_total.inc()
                return result
            except Exception as e:
                self.prediction_errors.inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def list_models():
            return self.list_models()
        
        @self.app.get("/metrics")
        async def metrics():
            return prometheus_client.generate_latest()
    
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any]) -> str:
        """
        모델 저장
        
        Args:
            model: 저장할 모델 객체
            model_name: 모델 이름
            metadata: 모델 메타데이터
            
        Returns:
            모델 버전
        """
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"{model_name}_{version}")
        
        # 모델 저장
        os.makedirs(model_path, exist_ok=True)
        
        # PyTorch 모델인 경우
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
        # scikit-learn 모델인 경우
        else:
            joblib.dump(model, os.path.join(model_path, "model.joblib"))
        
        # 메타데이터 저장
        with open(os.path.join(model_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        # 캐시 업데이트
        self.model_cache[f"{model_name}_{version}"] = model
        self.model_metadata[f"{model_name}_{version}"] = metadata
        
        # 메트릭스 업데이트
        self.model_versions.set(len(self.model_cache))
        
        return version
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        모델 로드
        
        Args:
            model_name: 모델 이름
            version: 모델 버전 (None이면 최신 버전)
            
        Returns:
            (모델 객체, 메타데이터)
        """
        if version is None:
            # 최신 버전 찾기
            versions = [v for v in self.model_metadata.keys() if v.startswith(model_name)]
            if not versions:
                raise ValueError(f"No models found for {model_name}")
            version = max(versions)
        
        model_key = f"{model_name}_{version}"
        
        # 캐시에서 로드
        if model_key in self.model_cache:
            return self.model_cache[model_key], self.model_metadata[model_key]
        
        # 디스크에서 로드
        model_path = os.path.join(self.model_dir, model_key)
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_key} not found")
        
        # 모델 로드
        model_file = os.path.join(model_path, "model.pt")
        if os.path.exists(model_file):
            model = torch.load(model_file)
        else:
            model = joblib.load(os.path.join(model_path, "model.joblib"))
        
        # 메타데이터 로드
        with open(os.path.join(model_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # 캐시에 저장
        self.model_cache[model_key] = model
        self.model_metadata[model_key] = metadata
        
        return model, metadata
    
    def predict(self, data: List[float], model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        실시간 예측 수행
        
        Args:
            data: 입력 데이터
            model_version: 모델 버전
            
        Returns:
            예측 결과
        """
        try:
            # TODO: 모델 선택 로직 구현
            model, metadata = self.load_model("default", model_version)
            
            # 예측 수행
            forecast, lower, upper = model.predict(n_forecast=1)
            
            # 이상치 탐지
            detector = BayesianAnomalyDetector()
            anomaly_score = detector.detect_anomaly(data[-1], forecast[0])
            
            return {
                "forecast": forecast[0],
                "lower_bound": lower[0],
                "upper_bound": upper[0],
                "anomaly_score": anomaly_score,
                "model_version": model_version,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def start(self):
        """배포 시스템 시작"""
        self.is_running = True
        
        # 예측 스레드 시작
        self.prediction_thread = threading.Thread(target=self._prediction_worker)
        self.prediction_thread.start()
        
        # 모니터링 스레드 시작
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker)
        self.monitoring_thread.start()
        
        # API 서버 시작
        uvicorn.run(self.app, host=self.api_host, port=self.api_port)
    
    def stop(self):
        """배포 시스템 중지"""
        self.is_running = False
        
        if self.prediction_thread:
            self.prediction_thread.join()
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _prediction_worker(self):
        """예측 워커 스레드"""
        while self.is_running:
            try:
                # 큐에서 예측 요청 가져오기
                request = self.prediction_queue.get(timeout=1)
                if request is None:
                    continue
                
                # 예측 수행
                result = self.predict(request["data"], request.get("model_version"))
                
                # 결과 전송
                if "callback" in request:
                    request["callback"](result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Prediction worker error: {str(e)}")
    
    def _monitoring_worker(self):
        """모니터링 워커 스레드"""
        while self.is_running:
            try:
                # 모델 성능 모니터링
                for model_key, model in self.model_cache.items():
                    # TODO: 성능 메트릭스 계산
                    pass
                
                # 메모리 사용량 모니터링
                self.model_memory_usage.set(self._get_memory_usage())
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring worker error: {str(e)}")
    
    def _get_memory_usage(self) -> int:
        """모델 메모리 사용량 계산"""
        total_memory = 0
        for model in self.model_cache.values():
            if isinstance(model, torch.nn.Module):
                total_memory += sum(p.numel() * p.element_size() for p in model.parameters())
            else:
                # TODO: 다른 모델 타입의 메모리 사용량 계산
                pass
        return total_memory
    
    def list_models(self) -> List[Dict[str, Any]]:
        """저장된 모델 목록 반환"""
        models = []
        for model_key, metadata in self.model_metadata.items():
            model_name, version = model_key.split("_", 1)
            models.append({
                "name": model_name,
                "version": version,
                "metadata": metadata,
                "created_at": metadata.get("created_at", ""),
                "performance": metadata.get("performance", {})
            })
        return models 