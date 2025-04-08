사용자 가이드
===========

시스템 시작하기
-------------

1. 데이터 수집기 시작:

   .. code-block:: bash

      python src/collector/run.py

   데이터 수집기는 다음 정보를 수집합니다:
   
   * 실시간 가격 데이터
   * 거래량 데이터
   * 기술적 지표
   * 시장 심리 지표

2. 트레이딩 시스템 시작:

   .. code-block:: bash

      python src/trading/run.py

   트레이딩 시스템은 다음 기능을 수행합니다:
   
   * 매매 신호 생성
   * 포지션 관리
   * 리스크 관리
   * 주문 실행

3. 웹 대시보드 접속:

   .. code-block:: bash

      python src/dashboard/run.py

   브라우저에서 http://localhost:5000 접속

전략 설정
--------

1. 전략 파라미터 설정:

   .. code-block:: python

      strategy_params = {
          'symbol': 'BTC/USDT',
          'timeframe': '1h',
          'rsi_period': 14,
          'rsi_overbought': 70,
          'rsi_oversold': 30,
          'macd_fast': 12,
          'macd_slow': 26,
          'macd_signal': 9,
          'stop_loss': 0.02,
          'take_profit': 0.04
      }

2. 리스크 관리 설정:

   .. code-block:: python

      risk_params = {
          'max_position_size': 0.1,  # 계좌 잔고의 10%
          'max_daily_loss': 0.02,    # 일일 최대 손실 2%
          'max_drawdown': 0.1,       # 최대 드로다운 10%
          'trailing_stop': 0.01      # 트레일링 스탑 1%
      }

3. 알림 설정:

   .. code-block:: python

      notification_params = {
          'telegram_chat_id': 'YOUR_CHAT_ID',
          'notify_trades': True,
          'notify_errors': True,
          'notify_performance': True
      }

백테스팅
--------

1. 백테스팅 실행:

   .. code-block:: python

      from src.backtesting import Backtester
      
      backtester = Backtester(
          strategy=strategy,
          data=data,
          initial_capital=10000,
          commission=0.001
      )
      
      results = backtester.run()

2. 결과 분석:

   .. code-block:: python

      print(f"총 수익률: {results.total_return:.2%}")
      print(f"승률: {results.win_rate:.2%}")
      print(f"샤프 비율: {results.sharpe_ratio:.2f}")
      print(f"최대 드로다운: {results.max_drawdown:.2%}")

성과 분석
--------

1. 성과 지표 확인:

   * 웹 대시보드의 "성과 분석" 섹션에서 확인
   * 텔레그램으로 일일 성과 보고서 수신
   * CSV/Excel 형식으로 성과 데이터 내보내기

2. 차트 분석:

   * 자본 곡선
   * 수익률 분포
   * 드로다운 차트
   * 포지션별 성과 비교

3. 리스크 분석:

   * 변동성 분석
   * 상관관계 분석
   * VaR (Value at Risk) 계산
   * 스트레스 테스트

모니터링 및 관리
--------------

1. 실시간 모니터링:

   * 포지션 상태
   * 계좌 잔고
   * 수익/손실
   * 리스크 지표

2. 알림 관리:

   * 거래 알림
   * 에러 알림
   * 성과 보고
   * 리스크 경고

3. 설정 관리:

   * 전략 파라미터 조정
   * 리스크 한도 설정
   * 알림 설정 변경
   * API 키 관리

문제 해결
--------

1. 일반적인 문제:

   * 데이터 수집 지연
   * 주문 실행 실패
   * 알림 전송 실패
   * 성능 저하

2. 해결 방법:

   * 로그 확인
   * 시스템 상태 점검
   * 설정 검증
   * 네트워크 연결 확인

3. 지원 받기:

   * 이슈 트래커 사용
   * 문서 참조
   * 커뮤니티 포럼
   * 개발자 연락 