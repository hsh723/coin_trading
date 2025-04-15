# 프로그램 설치 및 실행 가이드

## 1. 시스템 요구사항
- Python 3.9 이상
- pip (Python 패키지 관리자)
- Git (선택사항)
- 가상환경 (권장)

## 2. 프로젝트 다운로드
### Git을 사용하는 경우
```bash
git clone https://github.com/yourusername/coin_Trading.git
cd coin_Trading
```

### 직접 다운로드하는 경우
1. 프로젝트 ZIP 파일 다운로드
2. 원하는 위치에 압축 해제
3. 터미널에서 프로젝트 디렉토리로 이동

## 3. 가상환경 설정
### Windows
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\activate
```

### Linux/Mac
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

## 4. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 5. 환경 설정
1. `src/config/settings.py` 파일 열기
2. 다음 설정값 확인 및 수정:
   - `API_KEY`: 바이낸스 API 키
   - `API_SECRET`: 바이낸스 API 시크릿
   - `TEST_MODE`: 테스트 모드 여부 (True/False)
   - `LOG_LEVEL`: 로그 레벨 (DEBUG/INFO/WARNING/ERROR)
   - `DATABASE_URL`: 데이터베이스 연결 URL (필요한 경우)

## 6. API 키 설정
1. 바이낸스 계정 생성 (없는 경우)
2. API 키 생성:
   - 바이낸스 계정 로그인
   - API 관리 페이지로 이동
   - 새 API 키 생성
   - 필요한 권한 선택 (주문, 계정 정보 등)
3. 생성된 API 키와 시크릿을 `settings.py`에 입력

## 7. 데이터베이스 설정 (필요한 경우)
1. 데이터베이스 서버 설치 (예: PostgreSQL)
2. 데이터베이스 생성
3. `settings.py`에서 `DATABASE_URL` 설정
4. 필요한 테이블 생성 (프로그램이 자동 생성)

## 8. 로깅 설정
1. `src/utils/logger.py` 파일 확인
2. 로그 레벨 및 출력 위치 설정
3. 로그 파일 경로 확인 및 필요시 수정

## 9. 테스트 실행 (선택사항)
```bash
# 모든 테스트 실행
pytest

# 특정 테스트 실행
pytest tests/integration/test_binance_integration.py
pytest tests/execution/test_execution_manager.py
```

## 10. 프로그램 실행
```bash
# 메인 프로그램 실행
python src/main.py
```

## 11. API 접근
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 12. 모니터링
1. 로그 파일 확인
2. API 응답 확인
3. 주문 실행 상태 모니터링
4. 리스크 관리 상태 확인

## 13. 문제 해결
### 일반적인 문제
1. API 연결 실패
   - API 키와 시크릿 확인
   - 인터넷 연결 확인
   - 바이낸스 서버 상태 확인

2. 주문 실행 실패
   - 계정 잔고 확인
   - 주문 파라미터 검증
   - 시장 상태 확인

3. 데이터베이스 연결 실패
   - 연결 URL 확인
   - 데이터베이스 서버 상태 확인
   - 권한 설정 확인

### 로그 확인
- 로그 파일 위치: `logs/`
- 주요 로그 파일:
  - `app.log`: 애플리케이션 로그
  - `error.log`: 에러 로그
  - `execution.log`: 주문 실행 로그

## 14. 종료
1. 프로그램 정상 종료:
   - 터미널에서 Ctrl+C 입력
   - 모든 리소스 정리 확인

2. 비정상 종료 시:
   - 로그 파일 확인
   - 미처리된 주문 확인
   - 리소스 정리

## 15. 유지보수
1. 정기적인 로그 확인
2. API 키 주기적 갱신
3. 데이터베이스 백업
4. 시스템 업데이트 확인

## 16. 보안 주의사항
1. API 키와 시크릿은 절대 공개하지 않기
2. 정기적인 비밀번호 변경
3. IP 제한 설정 사용
4. 불필요한 API 권한 제거

## 17. 성능 최적화
1. 캐시 설정 확인
2. 데이터베이스 인덱스 최적화
3. 로그 레벨 조정
4. 리소스 사용량 모니터링 