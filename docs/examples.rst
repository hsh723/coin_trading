예제 코드
========

기본 전략 구현
------------

RSI + MACD 전략:

.. code-block:: python

   from src.trading.strategy import BaseStrategy
   from src.trading.executor import OrderExecutor
   from src.trading.risk import RiskManager
   
   class RsiMacdStrategy(BaseStrategy):
       def __init__(self, params):
           super().__init__(params)
           self.rsi_period = params.get('rsi_period', 14)
           self.rsi_overbought = params.get('rsi_overbought', 70)
           self.rsi_oversold = params.get('rsi_oversold', 30)
           self.macd_fast = params.get('macd_fast', 12)
           self.macd_slow = params.get('macd_slow', 26)
           self.macd_signal = params.get('macd_signal', 9)
   
       def generate_signals(self, data):
           signals = []
           
           # RSI 계산
           rsi = data['rsi'].iloc[-1]
           
           # MACD 계산
           macd = data['macd'].iloc[-1]
           macd_signal = data['macd_signal'].iloc[-1]
           
           # 매수 신호
           if rsi < self.rsi_oversold and macd > macd_signal:
               signals.append({
                   'symbol': self.symbol,
                   'direction': 'buy',
                   'price': data['close'].iloc[-1],
                   'time': data.index[-1]
               })
           
           # 매도 신호
           elif rsi > self.rsi_overbought and macd < macd_signal:
               signals.append({
                   'symbol': self.symbol,
                   'direction': 'sell',
                   'price': data['close'].iloc[-1],
                   'time': data.index[-1]
               })
           
           return signals

백테스팅 실행
-----------

.. code-block:: python

   from src.backtesting import Backtester
   from src.collector import DataCollector
   from src.analysis.performance import PerformanceAnalyzer
   
   # 데이터 수집
   collector = DataCollector()
   data = collector.get_historical_data(
       symbol='BTC/USDT',
       timeframe='1h',
       start_time='2023-01-01',
       end_time='2023-12-31'
   )
   
   # 전략 파라미터 설정
   strategy_params = {
       'symbol': 'BTC/USDT',
       'timeframe': '1h',
       'rsi_period': 14,
       'rsi_overbought': 70,
       'rsi_oversold': 30,
       'macd_fast': 12,
       'macd_slow': 26,
       'macd_signal': 9
   }
   
   # 백테스팅 실행
   backtester = Backtester(
       strategy=RsiMacdStrategy(strategy_params),
       data=data,
       initial_capital=10000,
       commission=0.001
   )
   
   results = backtester.run()
   
   # 성과 분석
   analyzer = PerformanceAnalyzer()
   metrics = analyzer.calculate_metrics(results.trades)
   
   print(f"총 수익률: {metrics['total_return']:.2%}")
   print(f"승률: {metrics['win_rate']:.2%}")
   print(f"샤프 비율: {metrics['sharpe_ratio']:.2f}")
   print(f"최대 드로다운: {metrics['max_drawdown']:.2%}")

실시간 트레이딩
-------------

.. code-block:: python

   from src.trading import TradingSystem
   from src.notification import TelegramNotifier
   
   # 트레이딩 시스템 초기화
   trading_system = TradingSystem(
       strategy=RsiMacdStrategy(strategy_params),
       risk_manager=RiskManager(risk_params),
       executor=OrderExecutor(executor_params)
   )
   
   # 알림 시스템 설정
   notifier = TelegramNotifier(
       bot_token='YOUR_BOT_TOKEN',
       chat_id='YOUR_CHAT_ID'
   )
   
   # 트레이딩 시작
   trading_system.start()
   
   try:
       while True:
           # 실시간 데이터 수집
           data = collector.get_realtime_data(
               symbol='BTC/USDT',
               timeframe='1h'
           )
           
           # 매매 신호 생성
           signals = trading_system.generate_signals(data)
           
           # 신호 실행
           for signal in signals:
               if trading_system.check_risk(signal):
                   order = trading_system.execute_order(signal)
                   notifier.send_trade_notification(order)
           
           time.sleep(60)  # 1분 대기
   
   except KeyboardInterrupt:
       trading_system.stop()
       print("트레이딩 시스템 종료")

웹 대시보드 커스터마이징
---------------------

.. code-block:: python

   from src.dashboard import DashboardApp
   from src.dashboard.routes import register_routes
   
   # 대시보드 앱 초기화
   app = DashboardApp()
   
   # 커스텀 라우트 등록
   @app.route('/api/custom_metrics')
   def get_custom_metrics():
       return {
           'custom_metric1': calculate_metric1(),
           'custom_metric2': calculate_metric2()
       }
   
   # 차트 데이터 커스터마이징
   @app.route('/api/custom_chart')
   def get_custom_chart():
       data = {
           'x': get_time_data(),
           'y': get_custom_data(),
           'type': 'scatter',
           'name': '커스텀 지표'
       }
       return data
   
   # 대시보드 실행
   app.run(host='0.0.0.0', port=5000)

성과 보고서 생성
-------------

.. code-block:: python

   from src.analysis.reporting import ReportGenerator
   
   # 보고서 생성기 초기화
   report_generator = ReportGenerator()
   
   # HTML 보고서 생성
   html_report = report_generator.generate_html_report(
       trades=results.trades,
       metrics=metrics,
       charts={
           'equity_curve': results.equity_curve,
           'drawdown': results.drawdown,
           'returns': results.returns
       }
   )
   
   # 텔레그램 보고서 생성
   telegram_report = report_generator.generate_telegram_report(
       trades=results.trades,
       metrics=metrics
   )
   
   # 보고서 저장
   with open('report.html', 'w') as f:
       f.write(html_report)
   
   # 텔레그램으로 보고서 전송
   notifier.send_report(telegram_report) 