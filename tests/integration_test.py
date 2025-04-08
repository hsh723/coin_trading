"""
통합 테스트 모듈
"""

import unittest
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.utils.error_handler import ErrorHandler
from src.utils.scaling_manager import ScalingManager
from src.utils.monitoring_dashboard import MonitoringDashboard
from src.strategy.portfolio_manager import PortfolioManager
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.analysis.pattern_recognizer import PatternRecognizer
from src.ai.prediction_model import PricePredictionModel
from src.ai.market_classifier import MarketClassifier

class IntegrationTest(unittest.TestCase):
    """통합 테스트 클래스"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 초기 설정"""
        cls.logger = logging.getLogger(__name__)
        cls.db_manager = None  # 실제 데이터베이스 연결 설정
        cls.error_handler = ErrorHandler(cls.db_manager)
        cls.scaling_manager = ScalingManager(cls.db_manager)
        cls.monitoring_dashboard = MonitoringDashboard(cls.db_manager)
        cls.portfolio_manager = PortfolioManager(cls.db_manager)
        cls.performance_analyzer = PerformanceAnalyzer(cls.db_manager)
        cls.pattern_recognizer = PatternRecognizer(cls.db_manager)
        cls.prediction_model = PricePredictionModel()
        cls.market_classifier = MarketClassifier()
        
    def test_error_handling(self):
        """에러 처리 테스트"""
        try:
            # API 에러 처리 테스트
            @self.error_handler.handle_api_error
            def test_api_call():
                raise Exception("API 에러 테스트")
                
            with self.assertRaises(Exception):
                test_api_call()
                
            # 데이터 에러 처리 테스트
            @self.error_handler.handle_data_error
            def test_data_processing():
                raise ValueError("데이터 에러 테스트")
                
            with self.assertRaises(ValueError):
                test_data_processing()
                
        except Exception as e:
            self.logger.error(f"에러 처리 테스트 실패: {str(e)}")
            self.fail("에러 처리 테스트 실패")
            
    def test_scaling_system(self):
        """스케일링 시스템 테스트"""
        try:
            # 작업 추가 테스트
            task_added = self.scaling_manager.add_trading_task(
                symbol="BTCUSDT",
                strategy="OrderBlockStrategy",
                params={"param1": "value1"}
            )
            self.assertTrue(task_added)
            
            # 작업 상태 조회 테스트
            tasks = self.scaling_manager.get_all_tasks()
            self.assertIsInstance(tasks, list)
            
        except Exception as e:
            self.logger.error(f"스케일링 시스템 테스트 실패: {str(e)}")
            self.fail("스케일링 시스템 테스트 실패")
            
    def test_monitoring_system(self):
        """모니터링 시스템 테스트"""
        try:
            # 모니터링 시작
            self.monitoring_dashboard.start_monitoring()
            
            # 리소스 사용량 차트 생성 테스트
            resource_chart = self.monitoring_dashboard.get_resource_usage_chart()
            self.assertIsNotNone(resource_chart)
            
            # API 성능 차트 생성 테스트
            api_chart = self.monitoring_dashboard.get_api_performance_chart()
            self.assertIsNotNone(api_chart)
            
            # 시스템 상태 조회 테스트
            system_status = self.monitoring_dashboard.get_system_status()
            self.assertIsInstance(system_status, dict)
            
            # 모니터링 중지
            self.monitoring_dashboard.stop_monitoring()
            
        except Exception as e:
            self.logger.error(f"모니터링 시스템 테스트 실패: {str(e)}")
            self.fail("모니터링 시스템 테스트 실패")
            
    def test_portfolio_management(self):
        """포트폴리오 관리 테스트"""
        try:
            # 전략 추가 테스트
            strategy_id = self.portfolio_manager.add_strategy(
                name="TestStrategy",
                params={"param1": "value1"}
            )
            self.assertIsNotNone(strategy_id)
            
            # 가중치 업데이트 테스트
            self.portfolio_manager.update_weights(lookback_days=30)
            
            # 포트폴리오 상태 조회 테스트
            portfolio_status = self.portfolio_manager.get_portfolio_status()
            self.assertIsInstance(portfolio_status, dict)
            
        except Exception as e:
            self.logger.error(f"포트폴리오 관리 테스트 실패: {str(e)}")
            self.fail("포트폴리오 관리 테스트 실패")
            
    def test_performance_analysis(self):
        """성과 분석 테스트"""
        try:
            # 거래 분석 테스트
            trade_analysis = self.performance_analyzer.analyze_trades(
                start_time=datetime.now() - timedelta(days=7),
                end_time=datetime.now()
            )
            self.assertIsInstance(trade_analysis, dict)
            
            # 시간 기반 성과 분석 테스트
            time_analysis = self.performance_analyzer.analyze_time_based_performance()
            self.assertIsInstance(time_analysis, dict)
            
            # 성과 보고서 생성 테스트
            report = self.performance_analyzer.generate_performance_report()
            self.assertIsInstance(report, dict)
            
        except Exception as e:
            self.logger.error(f"성과 분석 테스트 실패: {str(e)}")
            self.fail("성과 분석 테스트 실패")
            
    def test_pattern_recognition(self):
        """패턴 인식 테스트"""
        try:
            # 캔들스틱 패턴 인식 테스트
            patterns = self.pattern_recognizer.identify_candlestick_patterns(
                data=pd.DataFrame()  # 테스트 데이터
            )
            self.assertIsInstance(patterns, dict)
            
            # 가격 패턴 인식 테스트
            price_patterns = self.pattern_recognizer.identify_price_patterns(
                data=pd.DataFrame()  # 테스트 데이터
            )
            self.assertIsInstance(price_patterns, dict)
            
        except Exception as e:
            self.logger.error(f"패턴 인식 테스트 실패: {str(e)}")
            self.fail("패턴 인식 테스트 실패")
            
    def test_ai_models(self):
        """AI 모델 테스트"""
        try:
            # 예측 모델 테스트
            prediction = self.prediction_model.predict(
                data=pd.DataFrame()  # 테스트 데이터
            )
            self.assertIsInstance(prediction, dict)
            
            # 시장 분류 테스트
            market_state = self.market_classifier.classify_market(
                data=pd.DataFrame()  # 테스트 데이터
            )
            self.assertIsInstance(market_state, dict)
            
        except Exception as e:
            self.logger.error(f"AI 모델 테스트 실패: {str(e)}")
            self.fail("AI 모델 테스트 실패")
            
    @classmethod
    def tearDownClass(cls):
        """테스트 종료 정리"""
        try:
            cls.scaling_manager.shutdown()
            cls.monitoring_dashboard.stop_monitoring()
            
        except Exception as e:
            cls.logger.error(f"테스트 종료 정리 실패: {str(e)}")
            
if __name__ == '__main__':
    unittest.main() 