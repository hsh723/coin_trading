기여하기
========

암호화폐 트레이딩 시스템에 기여하고 싶으시다면 감사합니다! 이 문서는 프로젝트에 기여하는 방법을 안내합니다.

개발 환경 설정
------------

1. 저장소 복제
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/username/crypto-trading.git
   cd crypto-trading

2. 가상환경 설정
~~~~~~~~~~~~~

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

3. 의존성 설치
~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt
   pip install -r requirements-dev.txt

4. 개발 서버 실행
~~~~~~~~~~~~~

.. code-block:: bash

   python src/dashboard/run.py

코드 스타일
---------

1. PEP 8 준수
~~~~~~~~~~

* 들여쓰기: 4칸
* 최대 줄 길이: 79자
* 빈 줄: 함수/클래스 사이에 2줄
* 임포트 순서: 표준 라이브러리 > 서드파티 > 로컬

2. 문서화
~~~~~~~

* 모든 함수/클래스에 독스트링 추가
* Google 스타일 독스트링 사용
* 예제 코드 포함
* 타입 힌트 사용

3. 테스트
~~~~~~~

* 모든 새로운 기능에 대한 테스트 작성
* 테스트 커버리지 80% 이상 유지
* 통합 테스트 포함
* 모의 객체 사용

기여 프로세스
-----------

1. 이슈 생성
~~~~~~~~~~

* 버그 리포트
* 기능 요청
* 문서 개선
* 성능 최적화

2. 브랜치 생성
~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name
   git checkout -b fix/your-fix-name
   git checkout -b docs/your-docs-name

3. 코드 작성
~~~~~~~~~~

* 기능 구현
* 테스트 작성
* 문서 업데이트
* 코드 리뷰 요청

4. 테스트 실행
~~~~~~~~~~~

.. code-block:: bash

   pytest
   pytest --cov=src tests/
   flake8 src/
   mypy src/

5. 커밋
~~~~~

.. code-block:: bash

   git add .
   git commit -m "feat: 새로운 기능 추가"
   git commit -m "fix: 버그 수정"
   git commit -m "docs: 문서 업데이트"

6. PR 생성
~~~~~~~~

* PR 설명 작성
* 관련 이슈 링크
* 변경 사항 요약
* 스크린샷 첨부

코드 리뷰
--------

1. 리뷰어 역할
~~~~~~~~~~~

* 코드 품질 검토
* 테스트 커버리지 확인
* 문서화 검토
* 성능 영향 분석

2. 리뷰 프로세스
~~~~~~~~~~~~~

* PR 검토
* 코멘트 작성
* 변경 요청
* 승인/거절

3. 피드백 반영
~~~~~~~~~~~

* 코멘트 검토
* 코드 수정
* 재검토 요청
* 승인 대기

문서화
-----

1. API 문서
~~~~~~~~~

* 함수/클래스 설명
* 파라미터 설명
* 반환값 설명
* 예제 코드

2. 사용자 가이드
~~~~~~~~~~~~~

* 설치 방법
* 사용 방법
* 설정 방법
* 문제 해결

3. 개발자 문서
~~~~~~~~~~~

* 아키텍처 설명
* 개발 가이드
* 테스트 가이드
* 배포 가이드

릴리즈 프로세스
-------------

1. 버전 관리
~~~~~~~~~~

* 시맨틱 버저닝
* CHANGELOG.md 업데이트
* 버전 태그 생성
* 릴리즈 노트 작성

2. 배포
~~~~~

* 테스트 환경 배포
* 스테이징 환경 배포
* 프로덕션 환경 배포
* 모니터링

3. 문서 업데이트
~~~~~~~~~~~~~

* API 문서 업데이트
* 사용자 가이드 업데이트
* 개발자 문서 업데이트
* 예제 코드 업데이트

지원
----

1. 이슈 관리
~~~~~~~~~~

* 이슈 분류
* 우선순위 설정
* 마일스톤 관리
* 진행 상황 추적

2. 커뮤니티
~~~~~~~~~

* 질문 답변
* 토론 참여
* 피드백 수집
* 기여자 관리

3. 문서화
~~~~~~~~

* FAQ 관리
* 문제 해결 가이드
* 튜토리얼 작성
* 블로그 포스트

연락처
-----

* 이메일: support@example.com
* GitHub: https://github.com/username/crypto-trading
* Discord: https://discord.gg/crypto-trading
* 텔레그램: @crypto_trading 