# 암호화폐 자동매매 시스템

## 프로젝트 소개
이 프로젝트는 바이낸스 거래소를 대상으로 한 암호화폐 자동매매 시스템입니다. 기술적 지표와 리스크 관리를 결합한 통합 전략을 사용하여 자동으로 매매를 수행합니다.

## 주요 기능
- 실시간 시장 데이터 수집 및 분석
- 통합 매매 전략 구현 (이동평균, RSI, MACD, 볼린저밴드 등)
- 리스크 관리 시스템
- 가상/실제 트레이딩 모드 지원
- Docker 기반 배포 지원

## 시스템 구조
```
src/
├── exchange/         # 거래소 연동 모듈
│   └── binance.py    # 바이낸스 거래소 연동
├── indicators/       # 기술적 지표 모듈
│   ├── basic.py      # 기본 지표 (MA, RSI, MACD 등)
│   └── advanced.py   # 고급 지표 (볼린저밴드, 일목균형표 등)
├── strategies/       # 매매 전략 모듈
│   ├── base.py       # 기본 전략 클래스
│   └── integrated.py # 통합 매매 전략
├── risk/            # 리스크 관리 모듈
│   └── manager.py    # 리스크 관리 시스템
├── utils/           # 유틸리티 모듈
│   └── logger.py     # 로깅 시스템
└── main.py          # 메인 실행 파일
```

## 설치 방법

### 로컬 환경
1. Python 3.9 이상 설치
2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### Docker 환경
```bash
docker-compose up -d
```

## 실행 방법

### 로컬 환경
```bash
# 가상 트레이딩 모드
python src/main.py --mode paper

# 실제 트레이딩 모드
python src/main.py --mode live
```

### Docker 환경
```bash
# 컨테이너 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 컨테이너 중지
docker-compose down
```

## 설정
- `src/main.py`에서 거래소 API 키 설정
- `src/risk/manager.py`에서 리스크 관리 파라미터 설정
- `src/strategies/integrated.py`에서 매매 전략 파라미터 설정

## 리스크 관리
- 포지션 한도: 최대 5개
- 거래당 리스크: 자본의 2%
- 일일 손실 한도: 자본의 5%
- 최대 드로다운: 15%

## 모니터링
- 로그 파일: `logs/trading_bot.log`
- 거래 기록: `data/trades.csv`
- 성능 지표: `data/performance.csv`

## 개발 현황
- [x] 기본 거래소 연동
- [x] 기술적 지표 구현
- [x] 통합 매매 전략 구현
- [x] 리스크 관리 시스템
- [x] 로깅 시스템
- [x] Docker 배포 지원
- [ ] 백테스팅 시스템
- [ ] 웹 대시보드
- [ ] 텔레그램 알림
- [ ] 성능 분석 리포트

## 향후 계획
1. 백테스팅 시스템 구현
2. 웹 기반 모니터링 대시보드 개발
3. 텔레그램 알림 시스템 추가
4. 성능 분석 리포트 자동화
5. AWS 클라우드 배포 지원

## 주의사항
- 실제 트레이딩 전에 반드시 가상 트레이딩으로 충분한 테스트를 진행하세요.
- API 키는 절대 공개되지 않도록 주의하세요.
- 리스크 관리 파라미터는 신중하게 설정하세요.

## 라이선스
MIT License 