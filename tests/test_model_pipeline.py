import pytest
from src.model.model_pipeline import ModelPipeline
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def model_pipeline():
    config_dir = "./config"
    pipeline_dir = "./pipelines"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(pipeline_dir, exist_ok=True)
    
    # 기본 설정 파일 생성
    config = {
        "default": {
            "pipeline": {
                "stages": [
                    {
                        "name": "data_loading",
                        "type": "data_loader",
                        "params": {
                            "source": "local",
                            "format": "csv"
                        }
                    },
                    {
                        "name": "preprocessing",
                        "type": "preprocessor",
                        "params": {
                            "scaling": "standard",
                            "imputation": "mean"
                        }
                    },
                    {
                        "name": "feature_engineering",
                        "type": "feature_engineer",
                        "params": {
                            "window_size": 10,
                            "indicators": ["rsi", "macd"]
                        }
                    },
                    {
                        "name": "model_training",
                        "type": "model_trainer",
                        "params": {
                            "model_type": "regression",
                            "epochs": 10
                        }
                    },
                    {
                        "name": "model_evaluation",
                        "type": "model_evaluator",
                        "params": {
                            "metrics": ["mse", "mae", "r2"]
                        }
                    }
                ],
                "scheduling": {
                    "interval": "daily",
                    "time": "00:00"
                },
                "monitoring": {
                    "metrics": ["execution_time", "memory_usage"],
                    "alert_threshold": 0.9
                }
            }
        }
    }
    with open(os.path.join(config_dir, "model_pipeline.json"), "w") as f:
        json.dump(config, f)
    
    return ModelPipeline(config_dir=config_dir, pipeline_dir=pipeline_dir)

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    n_samples = 1000
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=n_samples, freq="H"),
        "price": np.random.randn(n_samples) * 100 + 1000,
        "volume": np.random.randint(100, 1000, n_samples),
        "rsi": np.random.uniform(0, 100, n_samples),
        "macd": np.random.randn(n_samples)
    }
    return pd.DataFrame(data)

def test_model_pipeline_initialization(model_pipeline):
    assert model_pipeline is not None
    assert model_pipeline.config_dir == "./config"
    assert model_pipeline.pipeline_dir == "./pipelines"

def test_pipeline_creation(model_pipeline):
    # 파이프라인 생성 테스트
    pipeline = model_pipeline.create_pipeline()
    
    assert pipeline is not None
    assert os.path.exists(os.path.join(model_pipeline.pipeline_dir, "pipeline.json"))
    assert len(pipeline.stages) == 5

def test_stage_execution(model_pipeline, sample_data):
    # 단계 실행 테스트
    results = model_pipeline.execute_stage("preprocessing", sample_data)
    
    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert not results.isnull().any().any()

def test_pipeline_execution(model_pipeline, sample_data):
    # 전체 파이프라인 실행 테스트
    results = model_pipeline.execute_pipeline(sample_data)
    
    assert results is not None
    assert "preprocessed_data" in results
    assert "features" in results
    assert "model" in results
    assert "metrics" in results

def test_pipeline_scheduling(model_pipeline):
    # 파이프라인 스케줄링 테스트
    schedule = model_pipeline.schedule_pipeline()
    
    assert schedule is not None
    assert schedule.interval == "daily"
    assert schedule.time == "00:00"
    assert schedule.is_active()

def test_pipeline_monitoring(model_pipeline):
    # 파이프라인 모니터링 테스트
    metrics = model_pipeline.monitor_pipeline()
    
    assert metrics is not None
    assert "execution_time" in metrics
    assert "memory_usage" in metrics
    assert all(v >= 0 for v in metrics.values())

def test_pipeline_logging(model_pipeline, sample_data):
    # 파이프라인 로깅 테스트
    log_file = model_pipeline.log_execution(sample_data)
    
    assert log_file is not None
    assert os.path.exists(log_file)
    assert os.path.getsize(log_file) > 0

def test_pipeline_visualization(model_pipeline):
    # 파이프라인 시각화 테스트
    plot_file = model_pipeline.visualize_pipeline()
    
    assert plot_file is not None
    assert os.path.exists(plot_file)
    assert os.path.getsize(plot_file) > 0

def test_pipeline_performance(model_pipeline, sample_data):
    # 파이프라인 성능 테스트
    start_time = datetime.now()
    
    # 파이프라인 실행
    model_pipeline.execute_pipeline(sample_data)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 성능 기준: 1000개 샘플에 대한 파이프라인 실행을 30초 이내에 완료
    assert processing_time < 30.0

def test_error_handling(model_pipeline):
    # 에러 처리 테스트
    # 잘못된 단계 이름
    with pytest.raises(ValueError):
        model_pipeline.execute_stage("invalid_stage", pd.DataFrame())
    
    # 잘못된 데이터
    with pytest.raises(ValueError):
        model_pipeline.execute_pipeline(None)
    
    # 잘못된 스케줄 설정
    with pytest.raises(ValueError):
        model_pipeline.schedule_pipeline(interval="invalid")

def test_pipeline_configuration(model_pipeline):
    # 파이프라인 설정 테스트
    config = model_pipeline.get_configuration()
    
    assert config is not None
    assert "pipeline" in config
    assert "stages" in config["pipeline"]
    assert "scheduling" in config["pipeline"]
    assert "monitoring" in config["pipeline"]
    assert len(config["pipeline"]["stages"]) == 5
    assert "interval" in config["pipeline"]["scheduling"]
    assert "metrics" in config["pipeline"]["monitoring"] 