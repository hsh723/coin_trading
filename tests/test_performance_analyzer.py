"""
성능 분석기 테스트
"""

import pytest
import pytest_asyncio
import pandas as pd
from datetime import datetime, timedelta
import os
import shutil
from src.analysis.performance_analyzer import performance_analyzer
from src.utils.database import db_manager

@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """
    테스트를 위한 데이터베이스 설정
    """
    # 데이터베이스 파일 경로
    db_path = 'data/trading.db'
    
    # 기존 데이터베이스 파일 삭제
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # 데이터 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    
    # 데이터베이스 초기화
    await db_manager._init_database()
    
    yield
    
    # 테스트 후 데이터베이스 파일 삭제
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def sample_trade_data():
    """
    샘플 거래 데이터 생성
    """
    return {
        'id': 'test_trade_1',
        'symbol': 'BTC/USDT',
        'entry_time': datetime.now() - timedelta(hours=2),
        'exit_time': datetime.now(),
        'entry_price': 50000,
        'exit_price': 49000,
        'volume': 1.0,
        'side': 'buy',
        'status': 'closed',
        'profit_loss': -1000
    }

@pytest.fixture
def sample_market_data():
    """
    샘플 시장 데이터 생성
    """
    try:
        # 테스트 데이터 디렉토리 생성
        test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        # 샘플 데이터 생성
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        data = {
            'timestamp': dates,
            'open': pd.Series(pd.np.random.normal(50000, 1000, 100)),
            'high': pd.Series(pd.np.random.normal(51000, 1000, 100)),
            'low': pd.Series(pd.np.random.normal(49000, 1000, 100)),
            'close': pd.Series(pd.np.random.normal(50000, 1000, 100)),
            'volume': pd.Series(pd.np.random.normal(1000, 100, 100))
        }
        df = pd.DataFrame(data)
        
        # 데이터 저장
        df.to_csv(os.path.join(test_data_dir, 'market_data.csv'), index=False)
        return df
        
    except Exception as e:
        pytest.fail(f"샘플 데이터 생성 실패: {str(e)}")

@pytest.mark.asyncio
async def test_analyze_trade_failure(sample_trade_data):
    """
    거래 실패 분석 테스트
    """
    try:
        # 분석 실행
        result = await performance_analyzer.analyze_trade_failure(sample_trade_data)
        
        # 결과 검증
        assert isinstance(result, dict)
        assert 'failure_reasons' in result
        assert 'improvements' in result
        assert 'market_data' in result
        assert 'sentiment_score' in result
        assert 'technical_analysis' in result
        
        print("거래 실패 분석 테스트 성공")
        
    except Exception as e:
        print(f"거래 실패 분석 테스트 실패: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_run_backtest_with_analysis():
    """
    백테스트 실행 및 분석 테스트
    """
    try:
        # 테스트 기간 설정
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        # 백테스트 실행
        await performance_analyzer.run_backtest_with_analysis(start_date, end_date)
        
        print("백테스트 분석 테스트 성공")
        
    except Exception as e:
        print(f"백테스트 분석 테스트 실패: {str(e)}")
        raise

def test_analyze_moving_averages(sample_market_data):
    """
    이동평균선 분석 테스트
    """
    try:
        result = performance_analyzer._analyze_moving_averages(sample_market_data)
        
        assert isinstance(result, dict)
        assert 'trend_reversal' in result
        assert 'ma_short' in result
        assert 'ma_medium' in result
        assert 'ma_long' in result
        
        print("이동평균선 분석 테스트 성공")
        
    except Exception as e:
        print(f"이동평균선 분석 테스트 실패: {str(e)}")
        raise

def test_analyze_rsi(sample_market_data):
    """
    RSI 분석 테스트
    """
    try:
        result = performance_analyzer._analyze_rsi(sample_market_data)
        
        assert isinstance(result, dict)
        assert 'overbought' in result
        assert 'oversold' in result
        assert 'rsi_value' in result
        
        print("RSI 분석 테스트 성공")
        
    except Exception as e:
        print(f"RSI 분석 테스트 실패: {str(e)}")
        raise

def test_analyze_bollinger_bands(sample_market_data):
    """
    볼린저 밴드 분석 테스트
    """
    try:
        result = performance_analyzer._analyze_bollinger_bands(sample_market_data)
        
        assert isinstance(result, dict)
        assert 'price_outside_bands' in result
        assert 'upper_band' in result
        assert 'lower_band' in result
        assert 'band_width' in result
        
        print("볼린저 밴드 분석 테스트 성공")
        
    except Exception as e:
        print(f"볼린저 밴드 분석 테스트 실패: {str(e)}")
        raise

def test_analyze_volume(sample_market_data):
    """
    거래량 분석 테스트
    """
    try:
        result = performance_analyzer._analyze_volume(sample_market_data)
        
        assert isinstance(result, dict)
        assert 'low_volume' in result
        assert 'current_volume' in result
        assert 'volume_ma' in result
        
        print("거래량 분석 테스트 성공")
        
    except Exception as e:
        print(f"거래량 분석 테스트 실패: {str(e)}")
        raise

def test_generate_analysis_report():
    """
    분석 리포트 생성 테스트
    """
    try:
        # 샘플 분석 결과 생성
        analysis_result = {
            'timestamp': datetime.now(),
            'trade_id': 'test_trade_1',
            'failure_reasons': ['추세 전환 감지 실패'],
            'improvements': ['이동평균선 기간 조정'],
            'market_data': {},
            'sentiment_score': 0.5,
            'technical_analysis': {}
        }
        
        # 리포트 생성
        report_path = performance_analyzer.generate_analysis_report(analysis_result)
        
        assert isinstance(report_path, str)
        assert os.path.exists(report_path)
        
        # 리포트 내용 확인
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '거래 분석 리포트' in content
            assert 'technical_indicators' in content
            assert 'sentiment_analysis' in content
            assert 'improvement_suggestions' in content
        
        print("분석 리포트 생성 테스트 성공")
        
    except Exception as e:
        print(f"분석 리포트 생성 테스트 실패: {str(e)}")
        raise

if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, '-v']) 