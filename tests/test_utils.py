import pytest
from src.utils.helpers import (
    format_number,
    calculate_percentage,
    generate_timestamp,
    validate_symbol,
    parse_timeframe,
    calculate_pnl,
    format_price,
    calculate_position_size,
    validate_order_type,
    calculate_risk_reward_ratio
)
import time
import datetime
import numpy as np
from src.utils.utils_manager import UtilsManager
import os
import json
import pandas as pd
from datetime import timedelta

def test_format_number():
    # 숫자 포맷팅 테스트
    assert format_number(1234.5678, 2) == "1,234.57"
    assert format_number(0.00012345, 6) == "0.000123"
    assert format_number(1000000, 0) == "1,000,000"
    assert format_number(-1234.5678, 2) == "-1,234.57"

def test_calculate_percentage():
    # 퍼센트 계산 테스트
    assert calculate_percentage(50, 200) == 25.0
    assert calculate_percentage(0, 100) == 0.0
    assert calculate_percentage(100, 100) == 100.0
    assert calculate_percentage(-50, 200) == -25.0

def test_generate_timestamp():
    # 타임스탬프 생성 테스트
    timestamp = generate_timestamp()
    assert isinstance(timestamp, int)
    assert timestamp > 0
    
    # 밀리초 단위 확인
    current_time = int(time.time() * 1000)
    assert abs(timestamp - current_time) < 1000

def test_validate_symbol():
    # 심볼 유효성 검사 테스트
    assert validate_symbol("BTCUSDT") is True
    assert validate_symbol("ETHUSDT") is True
    assert validate_symbol("BTC-USD") is False
    assert validate_symbol("INVALID") is False
    assert validate_symbol("") is False

def test_parse_timeframe():
    # 타임프레임 파싱 테스트
    assert parse_timeframe("1m") == 60
    assert parse_timeframe("5m") == 300
    assert parse_timeframe("15m") == 900
    assert parse_timeframe("1h") == 3600
    assert parse_timeframe("4h") == 14400
    assert parse_timeframe("1d") == 86400
    assert parse_timeframe("1w") == 604800
    
    # 잘못된 타임프레임
    with pytest.raises(ValueError):
        parse_timeframe("invalid")

def test_calculate_pnl():
    # 손익 계산 테스트
    # 롱 포지션
    assert calculate_pnl(50000, 51000, 1.0, "long") == 1000.0
    assert calculate_pnl(50000, 49000, 1.0, "long") == -1000.0
    
    # 숏 포지션
    assert calculate_pnl(50000, 49000, 1.0, "short") == 1000.0
    assert calculate_pnl(50000, 51000, 1.0, "short") == -1000.0
    
    # 레버리지 적용
    assert calculate_pnl(50000, 51000, 1.0, "long", 10) == 10000.0
    assert calculate_pnl(50000, 49000, 1.0, "long", 10) == -10000.0

def test_format_price():
    # 가격 포맷팅 테스트
    assert format_price(1234.5678, "BTCUSDT") == "1,234.57"
    assert format_price(0.00012345, "BTCUSDT") == "0.000123"
    assert format_price(1000000, "BTCUSDT") == "1,000,000.00"
    assert format_price(-1234.5678, "BTCUSDT") == "-1,234.57"

def test_calculate_position_size():
    # 포지션 크기 계산 테스트
    account_balance = 10000.0
    risk_percentage = 0.02
    entry_price = 50000.0
    stop_loss = 49000.0
    
    position_size = calculate_position_size(
        account_balance,
        risk_percentage,
        entry_price,
        stop_loss
    )
    
    expected_size = (account_balance * risk_percentage) / (entry_price - stop_loss)
    assert abs(position_size - expected_size) < 0.0001

def test_validate_order_type():
    # 주문 타입 유효성 검사 테스트
    assert validate_order_type("market") is True
    assert validate_order_type("limit") is True
    assert validate_order_type("stop") is True
    assert validate_order_type("stop_limit") is True
    assert validate_order_type("invalid") is False
    assert validate_order_type("") is False

def test_calculate_risk_reward_ratio():
    # 리스크:리워드 비율 계산 테스트
    entry_price = 50000.0
    stop_loss = 49000.0
    take_profit = 52000.0
    
    rr_ratio = calculate_risk_reward_ratio(
        entry_price,
        stop_loss,
        take_profit
    )
    
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    expected_ratio = reward / risk
    
    assert abs(rr_ratio - expected_ratio) < 0.0001

def test_utility_performance():
    # 유틸리티 함수 성능 테스트
    start_time = time.time()
    
    # 여러 함수 반복 실행
    for _ in range(1000):
        format_number(np.random.random() * 10000, 2)
        calculate_percentage(np.random.random() * 100, np.random.random() * 100)
        generate_timestamp()
        validate_symbol("BTCUSDT")
        parse_timeframe("1h")
        calculate_pnl(50000, 51000, 1.0, "long")
        format_price(1234.5678, "BTCUSDT")
        calculate_position_size(10000, 0.02, 50000, 49000)
        validate_order_type("limit")
        calculate_risk_reward_ratio(50000, 49000, 52000)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 성능 기준: 1000회 실행이 1초 이내 완료
    assert execution_time < 1.0

@pytest.fixture
def utils_manager():
    return UtilsManager()

@pytest.fixture
def sample_data():
    # 샘플 데이터 생성
    prices = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "open": np.random.uniform(40000, 50000, 100),
        "high": np.random.uniform(50000, 60000, 100),
        "low": np.random.uniform(30000, 40000, 100),
        "close": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(100, 1000, 100)
    })
    
    trades = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=50, freq="30min"),
        "price": np.random.uniform(40000, 50000, 50),
        "quantity": np.random.uniform(0.1, 10, 50),
        "side": np.random.choice(["BUY", "SELL"], 50)
    })
    
    return prices, trades

def test_utils_manager_initialization(utils_manager):
    assert utils_manager is not None

def test_data_processing(utils_manager, sample_data):
    # 데이터 처리 테스트
    prices, trades = sample_data
    
    # 데이터 정규화
    normalized_prices = utils_manager.normalize_data(prices[["open", "high", "low", "close"]])
    assert normalized_prices is not None
    assert normalized_prices.shape == prices[["open", "high", "low", "close"]].shape
    
    # 데이터 스케일링
    scaled_prices = utils_manager.scale_data(prices[["open", "high", "low", "close"]])
    assert scaled_prices is not None
    assert scaled_prices.shape == prices[["open", "high", "low", "close"]].shape
    
    # 데이터 클리닝
    cleaned_prices = utils_manager.clean_data(prices)
    assert cleaned_prices is not None
    assert cleaned_prices.shape == prices.shape

def test_time_utils(utils_manager):
    # 시간 유틸리티 테스트
    # 타임스탬프 변환
    timestamp = int(datetime.now().timestamp() * 1000)
    dt = utils_manager.timestamp_to_datetime(timestamp)
    assert isinstance(dt, datetime)
    
    # 시간 문자열 변환
    time_str = "2023-01-01 00:00:00"
    dt = utils_manager.str_to_datetime(time_str)
    assert isinstance(dt, datetime)
    
    # 시간 차이 계산
    dt1 = datetime.now()
    dt2 = dt1 + timedelta(hours=1)
    diff = utils_manager.time_diff(dt1, dt2)
    assert diff == 3600  # 1시간 = 3600초

def test_math_utils(utils_manager):
    # 수학 유틸리티 테스트
    # 평균 계산
    data = [1, 2, 3, 4, 5]
    mean = utils_manager.calculate_mean(data)
    assert mean == 3.0
    
    # 표준편차 계산
    std = utils_manager.calculate_std(data)
    assert isinstance(std, float)
    
    # 백분위수 계산
    percentile = utils_manager.calculate_percentile(data, 50)
    assert percentile == 3.0

def test_file_utils(utils_manager):
    # 파일 유틸리티 테스트
    test_dir = "./test_data"
    test_file = os.path.join(test_dir, "test.txt")
    
    # 디렉토리 생성
    utils_manager.create_directory(test_dir)
    assert os.path.exists(test_dir)
    
    # 파일 저장
    data = {"test": "data"}
    utils_manager.save_to_file(test_file, data)
    assert os.path.exists(test_file)
    
    # 파일 로드
    loaded_data = utils_manager.load_from_file(test_file)
    assert loaded_data == data
    
    # 파일 삭제
    utils_manager.delete_file(test_file)
    assert not os.path.exists(test_file)
    
    # 디렉토리 삭제
    utils_manager.delete_directory(test_dir)
    assert not os.path.exists(test_dir)

def test_string_utils(utils_manager):
    # 문자열 유틸리티 테스트
    # 문자열 포맷팅
    formatted = utils_manager.format_string("Hello, {name}!", {"name": "World"})
    assert formatted == "Hello, World!"
    
    # 문자열 분할
    parts = utils_manager.split_string("BTC-USDT", "-")
    assert parts == ["BTC", "USDT"]
    
    # 문자열 결합
    joined = utils_manager.join_strings(["BTC", "USDT"], "-")
    assert joined == "BTC-USDT"

def test_validation_utils(utils_manager):
    # 검증 유틸리티 테스트
    # 숫자 검증
    assert utils_manager.is_number("123") is True
    assert utils_manager.is_number("abc") is False
    
    # 날짜 검증
    assert utils_manager.is_valid_date("2023-01-01") is True
    assert utils_manager.is_valid_date("2023-13-01") is False
    
    # 이메일 검증
    assert utils_manager.is_valid_email("test@example.com") is True
    assert utils_manager.is_valid_email("invalid-email") is False

def test_performance_utils(utils_manager):
    # 성능 유틸리티 테스트
    # 실행 시간 측정
    @utils_manager.measure_time
    def test_function():
        time.sleep(0.1)
    
    execution_time = test_function()
    assert isinstance(execution_time, float)
    assert execution_time >= 0.1
    
    # 메모리 사용량 측정
    memory_usage = utils_manager.measure_memory()
    assert isinstance(memory_usage, float)
    assert memory_usage > 0

def test_error_handling(utils_manager):
    # 에러 처리 테스트
    # 예외 발생
    with pytest.raises(Exception):
        utils_manager.raise_error("Test error")
    
    # 예외 로깅
    try:
        raise ValueError("Test error")
    except Exception as e:
        utils_manager.log_error(e)
    
    # 예외 재시도
    attempts = 0
    @utils_manager.retry_on_error(max_attempts=3)
    def failing_function():
        nonlocal attempts
        attempts += 1
        raise Exception("Test error")
    
    with pytest.raises(Exception):
        failing_function()
    assert attempts == 3

def test_config_utils(utils_manager):
    # 설정 유틸리티 테스트
    # 환경 변수 설정
    utils_manager.set_env_variable("TEST_VAR", "test_value")
    assert os.environ["TEST_VAR"] == "test_value"
    
    # 환경 변수 조회
    value = utils_manager.get_env_variable("TEST_VAR")
    assert value == "test_value"
    
    # 환경 변수 삭제
    utils_manager.delete_env_variable("TEST_VAR")
    assert "TEST_VAR" not in os.environ 