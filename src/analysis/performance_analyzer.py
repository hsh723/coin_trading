"""
성과 분석 모듈
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
from ..utils.logger import setup_logger
from src.visualization.advanced_charts import AdvancedCharts

# Windows 환경에서 이벤트 루프 정책 변경
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """성과 분석 클래스"""
    
    def __init__(self, db_manager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.charts = AdvancedCharts()
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

    def analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        거래 내역 분석
        
        Args:
            trades (List[Dict[str, Any]]): 거래 내역
            
        Returns:
            Dict[str, Any]: 거래 분석 결과
        """
        try:
            if not trades:
                return {}
                
            # 거래 데이터프레임 생성
            df = pd.DataFrame(trades)
            
            # 기본 통계
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 수익/손실 분석
            total_pnl = df['pnl'].sum()
            avg_pnl = df['pnl'].mean()
            max_pnl = df['pnl'].max()
            min_pnl = df['pnl'].min()
            
            # 승리/패배 거래 분석
            winning_pnls = df[df['pnl'] > 0]['pnl']
            losing_pnls = df[df['pnl'] < 0]['pnl']
            
            avg_win = winning_pnls.mean() if not winning_pnls.empty else 0
            avg_loss = losing_pnls.mean() if not losing_pnls.empty else 0
            max_win = winning_pnls.max() if not winning_pnls.empty else 0
            max_loss = losing_pnls.min() if not losing_pnls.empty else 0
            
            # 연속 승리/패배
            df['win'] = df['pnl'] > 0
            df['streak'] = (df['win'] != df['win'].shift()).cumsum()
            streaks = df.groupby('streak')['win'].agg(['count', 'first'])
            
            max_win_streak = streaks[streaks['first']]['count'].max()
            max_loss_streak = streaks[~streaks['first']]['count'].max()
            
            # 거래 기간 분석
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600  # 시간 단위
            
            avg_duration = df['duration'].mean()
            max_duration = df['duration'].max()
            min_duration = df['duration'].min()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'max_pnl': max_pnl,
                'min_pnl': min_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win': max_win,
                'max_loss': max_loss,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'avg_duration': avg_duration,
                'max_duration': max_duration,
                'min_duration': min_duration
            }
            
        except Exception as e:
            self.logger.error(f"거래 내역 분석 실패: {str(e)}")
            return {}
            
    def analyze_equity_curve(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        자본금 곡선 분석
        
        Args:
            equity_curve (List[Dict[str, Any]]): 자본금 곡선 데이터
            
        Returns:
            Dict[str, Any]: 자본금 곡선 분석 결과
        """
        try:
            if not equity_curve:
                return {}
                
            # 자본금 곡선 데이터프레임 생성
            df = pd.DataFrame(equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 수익률 계산
            df['returns'] = df['equity'].pct_change()
            
            # 총 수익률
            total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
            
            # 연간 수익률
            days = (df.index[-1] - df.index[0]).days
            annual_return = (1 + total_return) ** (365 / days) - 1
            
            # 최대 낙폭
            df['peak'] = df['equity'].cummax()
            df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
            max_drawdown = df['drawdown'].min()
            
            # 낙폭 기간
            df['drawdown_start'] = (df['drawdown'] == 0) & (df['drawdown'].shift(1) != 0)
            df['drawdown_end'] = (df['drawdown'] == 0) & (df['drawdown'].shift(-1) != 0)
            
            drawdown_periods = []
            start_idx = None
            
            for idx, row in df.iterrows():
                if row['drawdown_start']:
                    start_idx = idx
                elif row['drawdown_end'] and start_idx is not None:
                    drawdown_periods.append((start_idx, idx))
                    start_idx = None
            
            max_drawdown_duration = max(
                [(end - start).total_seconds() / (24 * 3600) for start, end in drawdown_periods]
            ) if drawdown_periods else 0
            
            # 변동성
            volatility = df['returns'].std() * np.sqrt(252)
            
            # 샤프 비율
            risk_free_rate = 0.02  # 연간 무위험 수익률 2%
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = df['returns'] - daily_risk_free
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 소르티노 비율
            downside_returns = df['returns'][df['returns'] < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = np.sqrt(252) * (df['returns'].mean() - daily_risk_free) / downside_std
            
            # 칼마 비율
            calmar_ratio = annual_return / abs(max_drawdown)
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            self.logger.error(f"자본금 곡선 분석 실패: {str(e)}")
            return {}
            
    def analyze_monthly_performance(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        월별 성과 분석
        
        Args:
            equity_curve (List[Dict[str, Any]]): 자본금 곡선 데이터
            
        Returns:
            Dict[str, Any]: 월별 성과 분석 결과
        """
        try:
            if not equity_curve:
                return {}
                
            # 자본금 곡선 데이터프레임 생성
            df = pd.DataFrame(equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 월별 수익률 계산
            df['month'] = df['timestamp'].dt.to_period('M')
            monthly_returns = df.groupby('month')['equity'].apply(
                lambda x: (x.iloc[-1] / x.iloc[0]) - 1
            )
            
            # 월별 통계
            monthly_stats = {
                'mean_return': monthly_returns.mean(),
                'median_return': monthly_returns.median(),
                'std_return': monthly_returns.std(),
                'positive_months': len(monthly_returns[monthly_returns > 0]),
                'negative_months': len(monthly_returns[monthly_returns < 0]),
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min()
            }
            
            # 월별 수익률 분포
            return_distribution = {
                '0-5%': len(monthly_returns[(monthly_returns >= 0) & (monthly_returns < 0.05)]),
                '5-10%': len(monthly_returns[(monthly_returns >= 0.05) & (monthly_returns < 0.10)]),
                '10-15%': len(monthly_returns[(monthly_returns >= 0.10) & (monthly_returns < 0.15)]),
                '15%+': len(monthly_returns[monthly_returns >= 0.15]),
                '-0-5%': len(monthly_returns[(monthly_returns < 0) & (monthly_returns >= -0.05)]),
                '-5-10%': len(monthly_returns[(monthly_returns < -0.05) & (monthly_returns >= -0.10)]),
                '-10-15%': len(monthly_returns[(monthly_returns < -0.10) & (monthly_returns >= -0.15)]),
                '-15%+': len(monthly_returns[monthly_returns < -0.15])
            }
            
            return {
                'monthly_stats': monthly_stats,
                'return_distribution': return_distribution
            }
            
        except Exception as e:
            self.logger.error(f"월별 성과 분석 실패: {str(e)}")
            return {}
            
    def generate_report(self, trades: List[Dict[str, Any]], equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        성과 리포트 생성
        
        Args:
            trades (List[Dict[str, Any]]): 거래 내역
            equity_curve (List[Dict[str, Any]]): 자본금 곡선 데이터
            
        Returns:
            Dict[str, Any]: 성과 리포트
        """
        try:
            # 거래 분석
            trade_analysis = self.analyze_trades(trades)
            
            # 자본금 곡선 분석
            equity_analysis = self.analyze_equity_curve(equity_curve)
            
            # 월별 성과 분석
            monthly_analysis = self.analyze_monthly_performance(equity_curve)
            
            # 리포트 생성
            report = {
                'trade_analysis': trade_analysis,
                'equity_analysis': equity_analysis,
                'monthly_analysis': monthly_analysis,
                'summary': {
                    'total_return': equity_analysis.get('total_return', 0),
                    'annual_return': equity_analysis.get('annual_return', 0),
                    'max_drawdown': equity_analysis.get('max_drawdown', 0),
                    'sharpe_ratio': equity_analysis.get('sharpe_ratio', 0),
                    'win_rate': trade_analysis.get('win_rate', 0),
                    'total_trades': trade_analysis.get('total_trades', 0),
                    'profit_factor': abs(trade_analysis.get('total_pnl', 0) / trade_analysis.get('avg_loss', 1))
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"성과 리포트 생성 실패: {str(e)}")
            return {}
            
    def compare_with_benchmark(self, equity_curve: List[Dict[str, Any]], benchmark: pd.Series) -> Dict[str, Any]:
        """
        벤치마크와의 비교 분석
        
        Args:
            equity_curve (List[Dict[str, Any]]): 자본금 곡선 데이터
            benchmark (pd.Series): 벤치마크 수익률
            
        Returns:
            Dict[str, Any]: 벤치마크 비교 분석 결과
        """
        try:
            # 자본금 곡선 데이터프레임 생성
            df = pd.DataFrame(equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 수익률 계산
            strategy_returns = df['equity'].pct_change()
            
            # 벤치마크와 일자 맞추기
            benchmark = benchmark.reindex(strategy_returns.index)
            
            # 상관관계
            correlation = strategy_returns.corr(benchmark)
            
            # 베타
            covariance = strategy_returns.cov(benchmark)
            benchmark_variance = benchmark.var()
            beta = covariance / benchmark_variance
            
            # 알파
            risk_free_rate = 0.02  # 연간 무위험 수익률 2%
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            alpha = (strategy_returns.mean() - daily_risk_free) - beta * (benchmark.mean() - daily_risk_free)
            
            # 정보 비율
            tracking_error = (strategy_returns - benchmark).std() * np.sqrt(252)
            information_ratio = (strategy_returns.mean() - benchmark.mean()) * np.sqrt(252) / tracking_error
            
            # 승률
            outperformance = (strategy_returns > benchmark).mean()
            
            return {
                'correlation': correlation,
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'outperformance': outperformance
            }
            
        except Exception as e:
            self.logger.error(f"벤치마크 비교 분석 실패: {str(e)}")
            return {}

# 전역 성능 분석기 인스턴스
performance_analyzer = PerformanceAnalyzer(database_manager) 