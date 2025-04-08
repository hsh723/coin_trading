# 프로젝트 구조

```
coin_Trading/
├── src/                          # 소스 코드 디렉토리
│   ├── data/                     # 데이터 관련 모듈
│   │   ├── collector.py          # 시장 데이터 수집기
│   │   └── processor.py          # 데이터 전처리 및 분석
│   │
│   ├── strategies/               # 트레이딩 전략
│   │   ├── base.py              # 기본 전략 클래스
│   │   ├── momentum.py          # 모멘텀 전략
│   │   ├── mean_reversion.py    # 평균 회귀 전략
│   │   ├── breakout.py          # 브레이크아웃 전략
│   │   └── orderblock_bos.py    # Order Block & BOS 전략
│   │
│   ├── execution/                # 주문 실행 모듈
│   │   ├── order_executor.py    # 주문 실행기
│   │   └── risk_manager.py      # 리스크 관리
│   │
│   ├── monitoring/               # 모니터링 시스템
│   │   ├── health_check.py      # 시스템 상태 점검
│   │   ├── resource_monitor.py  # 리소스 모니터링
│   │   ├── error_detector.py    # 오류 감지
│   │   ├── performance_tracker.py # 성능 추적
│   │   └── alert_system.py      # 알림 시스템
│   │
│   ├── utils/                    # 유틸리티 모듈
│   │   ├── logger.py            # 로깅 시스템
│   │   └── backup.py            # 백업 시스템
│   │
│   └── dashboard/                # 대시보드
│       ├── app.py               # 대시보드 애플리케이션
│       └── components/          # 대시보드 컴포넌트
│
├── tests/                        # 테스트 코드
│   ├── unit/                    # 단위 테스트
│   └── integration/             # 통합 테스트
│
├── docs/                         # 문서
│   ├── api_reference.md         # API 참조 문서
│   ├── strategies.md            # 전략 개발 가이드
│   ├── usage.md                 # 사용자 가이드
│   ├── troubleshooting.md       # 문제 해결 가이드
│   ├── maintenance_plan.md      # 유지보수 계획
│   ├── live_testing_plan.md     # 실전 테스트 계획
│   ├── optimization_strategies.md # 최적화 전략
│   └── project_structure.md     # 프로젝트 구조
│
├── config/                       # 설정 파일
│   ├── config.yaml              # 기본 설정
│   └── strategies.yaml          # 전략 설정
│
├── logs/                         # 로그 파일
├── backups/                      # 백업 파일
├── requirements.txt              # 의존성 목록
└── README.md                     # 프로젝트 설명
```

## 주요 컴포넌트 설명

### 1. 데이터 모듈 (`src/data/`)
- `collector.py`: 실시간 시장 데이터 수집
- `processor.py`: 데이터 전처리 및 기술적 지표 계산

### 2. 전략 모듈 (`src/strategies/`)
- `base.py`: 전략 기본 클래스
- `momentum.py`: 모멘텀 기반 전략
- `mean_reversion.py`: 평균 회귀 전략
- `breakout.py`: 브레이크아웃 전략
- `orderblock_bos.py`: Order Block & BOS 전략

### 3. 실행 모듈 (`src/execution/`)
- `order_executor.py`: 주문 실행 및 관리
- `risk_manager.py`: 리스크 관리 및 포지션 제어

### 4. 모니터링 시스템 (`src/monitoring/`)
- `health_check.py`: 시스템 상태 모니터링
- `resource_monitor.py`: 리소스 사용량 추적
- `error_detector.py`: 오류 감지 및 알림
- `performance_tracker.py`: 성능 메트릭 추적
- `alert_system.py`: 알림 시스템

### 5. 유틸리티 (`src/utils/`)
- `logger.py`: 로깅 시스템
- `backup.py`: 데이터 백업 및 복구

### 6. 대시보드 (`src/dashboard/`)
- `app.py`: 대시보드 메인 애플리케이션
- `components/`: 대시보드 UI 컴포넌트

### 7. 문서 (`docs/`)
- API 참조, 전략 가이드, 사용자 매뉴얼 등
- 유지보수, 테스트, 최적화 계획

### 8. 설정 (`config/`)
- 시스템 설정 파일
- 전략 파라미터 설정

## 주요 기능

1. **데이터 수집 및 처리**
   - 실시간 시장 데이터 수집
   - 기술적 지표 계산
   - 데이터 정규화 및 전처리

2. **트레이딩 전략**
   - 다양한 전략 구현
   - 전략 최적화
   - 백테스팅 지원

3. **주문 실행**
   - 자동 주문 실행
   - 리스크 관리
   - 포지션 관리

4. **모니터링 및 알림**
   - 시스템 상태 모니터링
   - 성능 추적
   - 오류 감지 및 알림

5. **데이터 관리**
   - 자동 백업
   - 로그 관리
   - 데이터 복구

6. **사용자 인터페이스**
   - 실시간 대시보드
   - 성능 분석
   - 설정 관리 