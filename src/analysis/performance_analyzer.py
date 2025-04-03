"""
성능 분석 및 개선 모듈
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import psutil
import os
import platform
import tempfile
import atexit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.analysis.news_analyzer import news_analyzer
from src.database.database_manager import database_manager
from src.utils.config_loader import get_config
import uuid

# Windows 환경에서 이벤트 루프 정책 변경
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    def __init__(self):
        """
        성능 분석기 초기화
        """
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.min_samples = 100
        self.last_training_time = None
        self.training_interval = timedelta(hours=24)
        
        # 기본 디렉토리 설정
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        
        # 필요한 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.config = get_config()
    
    def _validate_trade_data(self, trade_data: Dict[str, Any]) -> bool:
        """
        거래 데이터 검증
        
        Args:
            trade_data (Dict[str, Any]): 거래 데이터
            
        Returns:
            bool: 검증 결과
        """
        required_fields = [
            'entry_time', 'exit_time', 
            'entry_price', 'exit_price', 'volume',
            'side'
        ]
        return all(field in trade_data for field in required_fields)
    
    async def analyze_trade_failure(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        거래 실패 분석
        
        Args:
            trade_data (Dict[str, Any]): 거래 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 거래 데이터 검증
            if not self._validate_trade_data(trade_data):
                raise ValueError("잘못된 거래 데이터 형식")
            
            # 데이터베이스 연결 확인
            if not await database_manager.check_connection():
                raise ConnectionError("데이터베이스 연결 실패")
            
            # 성능 모니터링
            self._monitor_performance()
            
            # 거래 데이터 저장
            trade_data['id'] = str(uuid.uuid4())
            trade_data['status'] = 'failed'
            trade_data['profit_loss'] = (trade_data['exit_price'] - trade_data['entry_price']) / trade_data['entry_price']
            await database_manager.save_trade_data(trade_data)
            
            # 실패 원인 분석
            failure_reasons = []
            
            # 1. 가격 변동 분석
            price_change = trade_data['profit_loss']
            if abs(price_change) > self.config['trading']['stop_loss']:
                failure_reasons.append('스탑로스 발동')
            
            # 2. 시장 상황 분석
            market_data = await self._get_market_data(trade_data)
            technical_analysis = self._analyze_technical_indicators(market_data)
            
            if technical_analysis['trend_reversal']:
                failure_reasons.append('추세 전환')
            if technical_analysis['high_volatility']:
                failure_reasons.append('높은 변동성')
            if technical_analysis['overbought']:
                failure_reasons.append('과매수')
            if technical_analysis['oversold']:
                failure_reasons.append('과매도')
            
            # 3. 뉴스 분석
            sentiment_score = await news_analyzer.analyze_market_sentiment()
            if sentiment_score < -0.5:
                failure_reasons.append('부정적인 뉴스')
            
            # market_data를 딕셔너리로 변환 (Timestamp를 문자열로 변환)
            market_data_dict = market_data.reset_index()
            market_data_dict['timestamp'] = market_data_dict['timestamp'].astype(str)
            market_data_dict = market_data_dict.to_dict(orient='records')
            
            # 분석 결과 저장
            analysis_result = {
                'id': str(uuid.uuid4()),
                'trade_id': trade_data['id'],
                'analysis_type': 'failure_analysis',
                'results': {
                    'failure_reasons': failure_reasons,
                    'technical_analysis': technical_analysis,
                    'sentiment_score': sentiment_score,
                    'market_data': market_data_dict,
                    'improvements': self._suggest_improvements(failure_reasons)
                }
            }
            
            await database_manager.save_analysis_result(analysis_result)
            
            # 알림 전송
            await database_manager.send_notification(
                'trade_failure',
                f"거래 실패 분석 완료\n"
                f"실패 원인: {', '.join(failure_reasons)}"
            )
            
            return analysis_result['results']
            
        except Exception as e:
            logger.error(f"거래 실패 분석 중 오류 발생: {str(e)}")
            await database_manager.send_notification(
                'error',
                f"거래 실패 분석 중 오류 발생: {str(e)}"
            )
            return {'error': str(e)}
    
    async def _get_market_data(self, trade_data: Dict[str, Any]) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            trade_data (Dict[str, Any]): 거래 데이터
            
        Returns:
            pd.DataFrame: 시장 데이터
        """
        try:
            # 데이터 조회 기간 설정
            start_time = trade_data['entry_time'] - pd.Timedelta(hours=24)
            end_time = trade_data['exit_time']
            
            # 실제 구현에서는 데이터베이스나 API에서 데이터 조회
            dates = pd.date_range(start=start_time, end=end_time, freq='1h')
            
            data = {
                'timestamp': dates,
                'open': np.random.normal(50000, 1000, len(dates)),
                'high': np.random.normal(51000, 1000, len(dates)),
                'low': np.random.normal(49000, 1000, len(dates)),
                'close': np.random.normal(50000, 1000, len(dates)),
                'volume': np.random.normal(1000, 100, len(dates))
            }
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"시장 데이터 조회 중 오류 발생: {str(e)}")
            raise
    
    def _analyze_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        기술적 지표 분석
        
        Args:
            df (pd.DataFrame): 시장 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            results = {
                'trend_reversal': False,
                'high_volatility': False,
                'overbought': False,
                'oversold': False
            }
            
            # 이동평균선
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 추세 전환 확인
            if df['ma5'].iloc[-1] < df['ma20'].iloc[-1] and df['ma5'].iloc[-2] > df['ma20'].iloc[-2]:
                results['trend_reversal'] = True
            
            # 변동성 확인
            volatility = df['bb_std'].iloc[-1] / df['bb_middle'].iloc[-1]
            if volatility > 0.02:  # 2% 이상의 변동성
                results['high_volatility'] = True
            
            # 과매수/과매도 확인
            if df['rsi'].iloc[-1] > 70:
                results['overbought'] = True
            elif df['rsi'].iloc[-1] < 30:
                results['oversold'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"기술적 지표 분석 중 오류 발생: {str(e)}")
            raise
    
    def _analyze_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        이동평균선 분석
        
        Args:
            df (pd.DataFrame): 시장 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 단기/중기/장기 이동평균선 계산
            df['ma_short'] = df['close'].rolling(window=5).mean()
            df['ma_medium'] = df['close'].rolling(window=20).mean()
            df['ma_long'] = df['close'].rolling(window=50).mean()
            
            # 추세 전환 감지
            trend_reversal = (
                (df['ma_short'].iloc[-1] < df['ma_medium'].iloc[-1]) and
                (df['ma_short'].iloc[-2] > df['ma_medium'].iloc[-2])
            )
            
            return {
                'trend_reversal': trend_reversal,
                'ma_short': df['ma_short'].iloc[-1],
                'ma_medium': df['ma_medium'].iloc[-1],
                'ma_long': df['ma_long'].iloc[-1]
            }
        except Exception as e:
            logger.error(f"이동평균선 분석 중 오류 발생: {str(e)}")
            return {}
    
    def _analyze_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        RSI 분석
        
        Args:
            df (pd.DataFrame): 시장 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # RSI 계산
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 과매수/과매도 상태 확인
            overbought = rsi.iloc[-1] > 70
            oversold = rsi.iloc[-1] < 30
            
            return {
                'overbought': overbought,
                'oversold': oversold,
                'rsi_value': rsi.iloc[-1]
            }
        except Exception as e:
            logger.error(f"RSI 분석 중 오류 발생: {str(e)}")
            return {}
    
    def _analyze_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        볼린저 밴드 분석
        
        Args:
            df (pd.DataFrame): 시장 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 볼린저 밴드 계산
            df['ma'] = df['close'].rolling(window=20).mean()
            df['std'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['ma'] + (df['std'] * 2)
            df['lower_band'] = df['ma'] - (df['std'] * 2)
            
            # 밴드 이탈 확인
            price_outside_bands = (
                df['close'].iloc[-1] > df['upper_band'].iloc[-1] or
                df['close'].iloc[-1] < df['lower_band'].iloc[-1]
            )
            
            return {
                'price_outside_bands': price_outside_bands,
                'upper_band': df['upper_band'].iloc[-1],
                'lower_band': df['lower_band'].iloc[-1],
                'band_width': (df['upper_band'].iloc[-1] - df['lower_band'].iloc[-1]) / df['ma'].iloc[-1]
            }
        except Exception as e:
            logger.error(f"볼린저 밴드 분석 중 오류 발생: {str(e)}")
            return {}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        거래량 분석
        
        Args:
            df (pd.DataFrame): 시장 데이터
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 거래량 이동평균
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            # 거래량 감소 확인
            low_volume = df['volume'].iloc[-1] < df['volume_ma'].iloc[-1] * 0.8
            
            return {
                'low_volume': low_volume,
                'current_volume': df['volume'].iloc[-1],
                'volume_ma': df['volume_ma'].iloc[-1]
            }
        except Exception as e:
            logger.error(f"거래량 분석 중 오류 발생: {str(e)}")
            return {}
    
    def _suggest_improvements(self, failure_reasons: List[str]) -> List[str]:
        """
        개선 사항 제안
        
        Args:
            failure_reasons (List[str]): 실패 원인 목록
            
        Returns:
            List[str]: 개선 사항 목록
        """
        try:
            improvements = []
            
            improvement_map = {
                '스탑로스 발동': '스탑로스 수준 조정 필요',
                '추세 전환': '추세 전환 감지 로직 개선 필요',
                '높은 변동성': '변동성 대응 전략 수립 필요',
                '부정적인 뉴스': '뉴스 감지 민감도 조정 필요',
                '과매수': 'RSI 상단 임계값 조정 필요',
                '과매도': 'RSI 하단 임계값 조정 필요'
            }
            
            for reason in failure_reasons:
                if reason in improvement_map:
                    improvements.append(improvement_map[reason])
            
            return improvements
            
        except Exception as e:
            logger.error(f"개선 사항 제안 중 오류 발생: {str(e)}")
            return []
    
    def _monitor_performance(self) -> None:
        """
        성능 모니터링
        """
        try:
            # Windows에서는 interval 파라미터 필요
            if platform.system() == 'Windows':
                cpu_percent = psutil.Process().cpu_percent(interval=1.0)
            else:
                cpu_percent = psutil.Process().cpu_percent()
            
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            if memory_usage > 1000:  # 1GB 이상
                logger.warning(f"높은 메모리 사용량: {memory_usage:.2f}MB")
            
            if cpu_percent > 80:
                logger.warning(f"높은 CPU 사용량: {cpu_percent}%")
            
        except Exception as e:
            logger.error(f"성능 모니터링 중 오류 발생: {str(e)}")
    
    def generate_analysis_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        분석 리포트 생성
        
        Args:
            analysis_result (Dict[str, Any]): 분석 결과
            
        Returns:
            str: HTML 리포트 경로
        """
        try:
            # 임시 파일 생성
            temp_dir = tempfile.gettempdir()
            report_path = os.path.join(temp_dir, f"report_{analysis_result['trade_id']}.html")
            
            # 프로그램 종료 시 임시 파일 정리
            atexit.register(lambda: os.remove(report_path) if os.path.exists(report_path) else None)
            
            # HTML 리포트 생성
            report = f"""
            <html>
                <head>
                    <title>거래 분석 리포트</title>
                    <meta charset="utf-8">
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                    </style>
                </head>
                <body>
                    <h1>거래 분석 리포트</h1>
                    <div class="section">
                        <h2>실패 원인</h2>
                        <ul>
                            {''.join(f"<li>{reason}</li>" for reason in analysis_result.get('failure_reasons', []))}
                        </ul>
                    </div>
                    <div class="section">
                        <h2>기술적 지표 분석</h2>
                        <div id="technical_indicators">
                            <h3>이동평균선</h3>
                            <pre>{analysis_result.get('technical_analysis', {}).get('moving_averages', {})}</pre>
                            <h3>RSI</h3>
                            <pre>{analysis_result.get('technical_analysis', {}).get('rsi', {})}</pre>
                            <h3>볼린저 밴드</h3>
                            <pre>{analysis_result.get('technical_analysis', {}).get('bollinger_bands', {})}</pre>
                            <h3>거래량</h3>
                            <pre>{analysis_result.get('technical_analysis', {}).get('volume', {})}</pre>
                        </div>
                    </div>
                    <div class="section">
                        <h2>뉴스 감성 분석</h2>
                        <div id="sentiment_analysis">
                            <p>감성 점수: {analysis_result.get('sentiment_score', 0):.2f}</p>
                            <p>분석 시간: {analysis_result.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}</p>
                        </div>
                    </div>
                    <div class="section">
                        <h2>개선 제안</h2>
                        <div id="improvement_suggestions">
                            <ul>
                                {''.join(f"<li>{improvement}</li>" for improvement in analysis_result.get('improvements', []))}
                            </ul>
                        </div>
                    </div>
                </body>
            </html>
            """
            
            # 리포트 파일 저장 (UTF-8 인코딩 사용)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            return report_path
            
        except Exception as e:
            logger.error(f"리포트 생성 중 오류 발생: {str(e)}")
            return ""
    
    async def run_backtest_with_analysis(self, start_date: datetime, end_date: datetime) -> None:
        """
        백테스트 실행 및 분석
        
        Args:
            start_date (datetime): 시작일
            end_date (datetime): 종료일
        """
        try:
            # 백테스트 실행
            results = await self._run_backtest(start_date, end_date)
            
            # 실패한 거래 분석
            for trade in results['failed_trades']:
                analysis = await self.analyze_trade_failure(trade)
                await database_manager.save_analysis_result(analysis)
                
            # 전체 성능 분석
            performance_analysis = self._analyze_overall_performance(results)
            await database_manager.save_analysis_result(performance_analysis)
            
        except Exception as e:
            logger.error(f"백테스트 분석 중 오류 발생: {str(e)}")
    
    async def _run_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            start_date (datetime): 시작일
            end_date (datetime): 종료일
            
        Returns:
            Dict[str, Any]: 백테스트 결과
        """
        # 임시 구현
        return {
            'failed_trades': [],
            'total_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0
        }
    
    def _analyze_overall_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        전체 성능 분석
        
        Args:
            results (Dict[str, Any]): 백테스트 결과
            
        Returns:
            Dict[str, Any]: 성능 분석 결과
        """
        # 임시 구현
        return {
            'analysis_type': 'performance_analysis',
            'results': {
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'total_return': results['total_return']
            }
        }

# 전역 성능 분석기 인스턴스
performance_analyzer = PerformanceAnalyzer() 