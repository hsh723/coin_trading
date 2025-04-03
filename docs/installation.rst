설치 가이드
===========

환경 설정
--------

1. Python 가상환경 생성 및 활성화:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/Mac
      venv\Scripts\activate     # Windows

2. 필요한 패키지 설치:

   .. code-block:: bash

      pip install -r requirements.txt

3. 환경 변수 설정:

   .. code-block:: bash

      # .env 파일 생성
      cp .env.example .env
      
      # 환경 변수 편집
      nano .env

   필수 환경 변수:
   
   * ``BINANCE_API_KEY``: 바이낸스 API 키
   * ``BINANCE_API_SECRET``: 바이낸스 API 시크릿
   * ``TELEGRAM_BOT_TOKEN``: 텔레그램 봇 토큰
   * ``DATABASE_URL``: PostgreSQL 데이터베이스 URL
   * ``REDIS_URL``: Redis URL

데이터베이스 설정
--------------

1. PostgreSQL 설치:

   .. code-block:: bash

      # Ubuntu
      sudo apt-get install postgresql postgresql-contrib
      
      # macOS
      brew install postgresql
      
      # Windows
      # https://www.postgresql.org/download/windows/ 에서 설치

2. 데이터베이스 생성:

   .. code-block:: bash

      createdb crypto_trading
      psql crypto_trading

3. 스키마 초기화:

   .. code-block:: bash

      python src/database/init_db.py

Redis 설정
---------

1. Redis 설치:

   .. code-block:: bash

      # Ubuntu
      sudo apt-get install redis-server
      
      # macOS
      brew install redis
      
      # Windows
      # https://github.com/microsoftarchive/redis/releases 에서 설치

2. Redis 서버 시작:

   .. code-block:: bash

      # Linux/macOS
      sudo service redis-server start
      
      # Windows
      redis-server

시스템 실행
---------

1. 데이터 수집기 실행:

   .. code-block:: bash

      python src/collector/run.py

2. 트레이딩 시스템 실행:

   .. code-block:: bash

      python src/trading/run.py

3. 웹 대시보드 실행:

   .. code-block:: bash

      python src/dashboard/run.py

문제 해결
--------

1. 데이터베이스 연결 오류:

   * PostgreSQL 서비스가 실행 중인지 확인
   * 데이터베이스 URL이 올바른지 확인
   * 사용자 권한이 올바르게 설정되어 있는지 확인

2. Redis 연결 오류:

   * Redis 서버가 실행 중인지 확인
   * Redis URL이 올바른지 확인
   * 방화벽 설정 확인

3. API 연결 오류:

   * API 키와 시크릿이 올바른지 확인
   * API 제한에 도달하지 않았는지 확인
   * 네트워크 연결 상태 확인

4. 텔레그램 봇 오류:

   * 봇 토큰이 올바른지 확인
   * 봇이 활성화되어 있는지 확인
   * 봇 권한이 올바르게 설정되어 있는지 확인 