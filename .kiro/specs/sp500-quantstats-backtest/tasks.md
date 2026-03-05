# 구현 계획: S&P 500 QuantStats 백테스팅 시스템

## 개요

Pine Script v6 기반 37개 리딩 지표와 45개+ 확인 지표를 Python으로 재구현하고, S&P 500 지수 대상 모든 지표 조합의 Long/Short 백테스팅을 수행하는 시스템을 단계적으로 구현한다. 각 단계는 이전 단계 위에 점진적으로 빌드되며, 마지막에 전체 파이프라인을 연결한다.

## Tasks

- [x] 1. 프로젝트 구조 및 핵심 인터페이스 설정
  - [x] 1.1 프로젝트 디렉토리 구조 생성 및 requirements.txt 작성
    - `sp500_backtest/` 하위에 `data/`, `indicators/`, `engine/`, `results/`, `tests/`, `checkpoint/`, `output/reports/` 디렉토리 생성
    - 각 패키지에 `__init__.py` 파일 생성
    - requirements.txt에 yfinance, quantstats, pandas, numpy, pandas-ta, pyyaml, hypothesis, pytest 포함
    - _요구사항: 9.3, 9.4_
  - [x] 1.2 설정 파일(config.yaml) 및 설정 로더 구현
    - 설계 문서의 config.yaml 구조대로 기본 설정 파일 생성
    - config.yaml을 로딩하여 딕셔너리로 반환하는 로더 함수 구현
    - 명령줄 인자 없이 기본값으로 실행 가능하도록 구성
    - _요구사항: 9.1, 9.2_
  - [x] 1.3 핵심 데이터 모델 및 기본 인터페이스 정의 (`indicators/base.py`)
    - `IndicatorResult` dataclass 정의 (long_signal, short_signal: pd.Series)
    - `BaseIndicator` 추상 클래스 정의 (name, default_params, calculate)
    - `ConfirmationIndicator` 클래스 정의 (subtypes, subtype 파라미터 지원)
    - _요구사항: 2.2, 2.4, 3.2, 3.3_
  - [ ]\* 1.4 설정 파일 라운드 트립 속성 테스트 작성
    - **Property 16: 설정 파일 라운드 트립**
    - **검증: 요구사항 9.1**

- [x] 2. 데이터 수집 모듈 구현 (`data/fetcher.py`)
  - [x] 2.1 DataFetcher 클래스 구현
    - yfinance를 사용하여 S&P 500(^GSPC) 일봉 OHLCV 데이터 수집
    - 수집된 데이터의 시작일, 종료일, 총 거래일 수 콘솔 출력
    - 결측값(NaN) 전방 채움(forward fill) 처리
    - 네트워크 오류 시 3회 재시도 후 에러 메시지 출력 및 안전 종료
    - Volume 데이터 포함 수집
    - _요구사항: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]\* 2.2 NaN 전방 채움 완전성 속성 테스트 작성
    - **Property 19: NaN 전방 채움 완전성**
    - **검증: 요구사항 1.4**

- [x] 3. 체크포인트 - 데이터 수집 및 기본 구조 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의

- [x] 4. 지표 공통 유틸리티 구현 (`indicators/utils.py`)
  - [x] 4.1 공통 유틸리티 함수 구현
    - `ma()`: SMA, EMA, RMA, WMA, HMA, VWMA 이동평균 계산
    - `atr()`: Average True Range 계산
    - `crossover()`, `crossunder()`: 크로스오버/크로스언더 감지
    - `stoch()`: Stochastic %K 계산
    - `rsi()`: RSI 계산
    - `true_range()`: True Range 계산
    - 모든 함수는 pandas/numpy 벡터화 연산 사용
    - _요구사항: 10.4_

- [x] 5. 리딩 지표 구현 (`indicators/leading.py`)
  - [x] 5.1 트렌드 추종 리딩 지표 구현 (1차 배치: 12개)
    - Range Filter, RQK, Supertrend, Half Trend, Ichimoku Cloud, SuperIchi 구현
    - Chandelier Exit, Parabolic SAR, SSL Channel, Hull Suite, Donchian Trend Ribbon, Trendline Breakout 구현
    - 각 지표는 BaseIndicator를 상속하고 Pine Script 기본 파라미터 동일 설정
    - 각 지표는 Long/Short 시그널을 IndicatorResult로 반환
    - _요구사항: 2.1, 2.2, 2.3, 2.4_
  - [x] 5.2 오실레이터/모멘텀 리딩 지표 구현 (2차 배치: 13개)
    - TSI, TDFI, Stochastic, RSI, ROC, CCI, MACD 구현
    - B-Xtrender, Bull Bear Power Trend, DPO, BB Oscillator, Awesome Oscillator, Volatility Oscillator 구현
    - 각 지표는 BaseIndicator를 상속하고 Pine Script 기본 파라미터 동일 설정
    - _요구사항: 2.1, 2.2, 2.3, 2.4_
  - [x] 5.3 크로스/복합 리딩 지표 구현 (3차 배치: 12개)
    - 2 EMA Cross, 3 EMA Cross, VWAP 구현
    - DMI (ADX), Waddah Attar Explosion, Chaikin Money Flow, Vortex Index 구현
    - STC, Range Detector, HACOLT, Wolfpack Id, QQE Mod 구현
    - 각 지표는 BaseIndicator를 상속하고 Pine Script 기본 파라미터 동일 설정
    - _요구사항: 2.1, 2.2, 2.3, 2.4_
  - [ ]\* 5.4 지표 반환값 형식 불변성 속성 테스트 작성 (리딩 지표)
    - **Property 1: 지표 반환값 형식 불변성 (리딩 지표 부분)**
    - Hypothesis로 임의의 OHLCV DataFrame과 파라미터를 생성하여 모든 리딩 지표의 반환값 형식 검증
    - **검증: 요구사항 2.2**

- [x] 6. 확인 지표 구현 (`indicators/confirmation.py`)
  - [x] 6.1 리딩 지표 공유 로직 기반 확인 지표 구현 (1차 배치)
    - EMA Filter, 2 EMA Cross, 3 EMA Cross, Range Filter (Default/DW), RQK, SuperTrend 구현
    - Half Trend, Donchian Trend Ribbon, ROC, McGinley Dynamic, DPO 구현
    - ConfirmationIndicator를 상속하고 서브타입 지원
    - _요구사항: 3.1, 3.2, 3.3, 3.4_
  - [x] 6.2 서브타입 포함 확인 지표 구현 (2차 배치)
    - TSI (Signal Cross/Zero line cross), B-Xtrender (Short term/Short and Long term) 구현
    - Bull Bear Power Trend (Follow Trend/Without Trend), BB Oscillator (Entering/Exiting Band) 구현
    - Stochastic (CrossOver/OB&OS levels/%K above-below %D) 구현
    - RSI (RSI MA Cross/RSI Exits OB-OS/RSI Level), RSI MA Direction, RSI Limit, RSI MA Limit 구현
    - MACD (MACD Crossover/Zero line crossover) 구현
    - _요구사항: 3.1, 3.2, 3.3, 3.4_
  - [x] 6.3 나머지 확인 지표 구현 (3차 배치)
    - Ichimoku Cloud, SuperIchi, Trendline Breakout, Range Detector, HACOLT 구현
    - Chandelier Exit, CCI, Parabolic SAR, SSL Channel, Hull Suite 구현
    - Awesome Oscillator (Zero Line Cross/AC Zero Line Cross/AC Momentum Bar) 구현
    - DMI-ADX (Adx Only/Adx & +Di -Di/Advance), Waddah Attar Explosion 구현
    - Volatility Oscillator, Choppiness Index, Damiani Volatility (Simple/Threshold/10p Difference) 구현
    - Volume (volume above MA/Simple/Delta), Wolfpack Id, QQE Mod (Line/Bar/Line & Bar) 구현
    - Chaikin Money Flow, Vortex Indicator (Simple/Advance), Schaff Trend Cycle, VWAP 구현
    - _요구사항: 3.1, 3.2, 3.3, 3.4_
  - [ ]\* 6.4 지표 반환값 형식 불변성 속성 테스트 작성 (확인 지표)
    - **Property 1: 지표 반환값 형식 불변성 (확인 지표 부분)**
    - Hypothesis로 임의의 OHLCV DataFrame, 파라미터, 서브타입을 생성하여 모든 확인 지표의 반환값 형식 검증
    - **검증: 요구사항 3.2, 3.3**

- [x] 7. 체크포인트 - 지표 라이브러리 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의

- [x] 8. 조합 생성 엔진 구현 (`engine/combination.py`)
  - [x] 8.1 CombinationEngine 클래스 구현
    - `IndicatorCombination` dataclass 정의 (id, leading, leading_params, confirmations)
    - 37개 리딩 지표 × 확인 지표 0~N개 조합 생성 로직 구현
    - 서브타입이 있는 지표를 별도 조합으로 취급
    - 조합 수가 max_combinations 초과 시 max_confirmations 자동 축소
    - 각 조합에 고유 식별자 부여
    - 총 조합 수 콘솔 출력
    - _요구사항: 4.1, 4.2, 4.3, 4.4, 4.5_
  - [ ]\* 8.2 조합 생성 속성 테스트 작성
    - **Property 3: 조합 수 정확성** — L × Σ(k=0..N) C(C,k) 공식 검증
    - **검증: 요구사항 4.1, 4.2**
  - [ ]\* 8.3 조합 수 상한 및 ID 고유성 속성 테스트 작성
    - **Property 4: 조합 수 상한 제한** — 생성된 조합 수 ≤ max_combinations 검증
    - **Property 5: 조합 ID 고유성** — 모든 조합 ID 중복 없음 검증
    - **검증: 요구사항 4.4, 4.5**

- [x] 9. 시그널 생성기 구현 (`engine/signal.py`)
  - [x] 9.1 SignalGenerator 클래스 구현
    - 리딩 지표 Long + 모든 확인 지표 Long 확인 → Long 포지션 시그널 생성
    - 리딩 지표 Short + 모든 확인 지표 Short 확인 → Short 포지션 시그널 생성
    - Signal Expiry 로직: 리딩 시그널 발생 후 signal_expiry 캔들 이내 미확인 시 무효화
    - Alternate Signal 로직: 연속 동일 방향 시그널 필터링
    - 반환값: pd.Series (1=Long, -1=Short, 0=무포지션)
    - _요구사항: 6.1, 6.2, 6.3, 6.4_
  - [ ]\* 9.2 시그널 AND 조건 결합 속성 테스트 작성
    - **Property 2: 시그널 AND 조건 결합**
    - **검증: 요구사항 6.1, 6.2**
  - [ ]\* 9.3 Signal Expiry 및 Alternate Signal 속성 테스트 작성
    - **Property 7: Signal Expiry 무효화**
    - **Property 8: Alternate Signal 필터링**
    - **검증: 요구사항 6.3, 6.4**

- [x] 10. 백테스팅 엔진 구현 (`engine/backtest.py`)
  - [x] 10.1 BacktestEngine 클래스 구현
    - `BacktestResult` dataclass 정의 (combination_id, total_return, cagr, max_drawdown, sharpe_ratio, sortino_ratio, total_trades, win_rate, strategy_returns, positions)
    - 포지션 시그널 기반 벡터화 수익률 계산 (Long=+1, Short=-1, 무포지션=0)
    - 시그널 발생 다음 거래일부터 포지션 반영 (shift(1))
    - 포지션 변경 시 거래 비용 차감 옵션
    - _요구사항: 6.5, 6.6, 6.7_
  - [ ]\* 10.2 백테스팅 엔진 속성 테스트 작성
    - **Property 9: 포지션 값 불변성** — 모든 포지션 값 ∈ {-1, 0, 1}
    - **Property 10: 시그널 1일 지연** — positions[t] == signals[t-1]
    - **Property 11: 거래 비용의 단조 감소 효과** — 비용 0 수익률 ≥ 비용 양수 수익률
    - **검증: 요구사항 6.5, 6.6, 6.7**

- [x] 11. 체크포인트 - 엔진 모듈 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의

- [x] 12. 파라미터 최적화 엔진 구현 (`engine/optimizer.py`)
  - [x] 12.1 ParameterOptimizer 클래스 구현
    - `ParamSearchSpace` dataclass 정의 (name, min_val, max_val, step)
    - 그리드 서치(Grid Search) 및 랜덤 서치(Random Search) 방식 구현
    - 파라미터 탐색 범위와 스텝 크기를 설정 파일/딕셔너리로 정의 가능
    - 멀티프로세싱(concurrent.futures) 활용 병렬 백테스팅
    - 진행률(완료 조합 수 / 전체 조합 수) 및 ETA 콘솔 출력
    - _요구사항: 5.1, 5.2, 5.3, 5.4, 5.5, 10.1, 10.3_
  - [ ]\* 12.2 파라미터 탐색 범위 준수 속성 테스트 작성
    - **Property 6: 파라미터 탐색 범위 준수**
    - **검증: 요구사항 5.2**

- [x] 13. 지표 캐시 구현
  - [x] 13.1 IndicatorCache 클래스 구현
    - 캐시 키: (indicator_name, frozenset(params.items())) 튜플
    - `get_or_compute()` 메서드: 캐시 히트 시 반환, 미스 시 계산 후 캐싱
    - 동일 지표가 여러 조합에서 사용될 때 중복 계산 방지
    - _요구사항: 10.2_
  - [ ]\* 13.2 지표 캐시 멱등성 속성 테스트 작성
    - **Property 18: 지표 캐시 멱등성**
    - **검증: 요구사항 10.2**

- [x] 14. 체크포인트 기능 구현
  - [x] 14.1 Checkpoint 저장/로딩 구현
    - `Checkpoint` dataclass 정의 (completed_combinations, results, timestamp, total_combinations)
    - 중간 결과를 주기적으로(checkpoint_interval 간격) 저장
    - 중단 후 재시작 시 체크포인트에서 이어서 실행
    - _요구사항: 9.5_
  - [ ]\* 14.2 체크포인트 라운드 트립 속성 테스트 작성
    - **Property 17: 체크포인트 라운드 트립**
    - **검증: 요구사항 9.5**

- [x] 15. 결과 정렬 및 순위 모듈 구현 (`results/ranker.py`)
  - [x] 15.1 ResultRanker 클래스 구현
    - 총 수익률(Total Return) 기준 내림차순 정렬
    - 각 조합별 total_return, cagr, max_drawdown, sharpe_ratio, total_trades, win_rate 계산 및 결과 테이블 포함
    - 정렬 기준 선택 옵션 (total_return, cagr, sharpe_ratio)
    - 결과를 CSV 파일로 저장
    - 상위 N개(기본값: 20) 조합 요약 콘솔 출력
    - _요구사항: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ]\* 15.2 결과 정렬 및 완전성 속성 테스트 작성
    - **Property 12: 결과 정렬 순서** — 지정 기준 내림차순 검증
    - **Property 13: 결과 테이블 완전성** — 필수 컬럼 포함 검증
    - **Property 14: CSV 저장/로딩 라운드 트립** — 저장 후 로딩 동일성 검증
    - **검증: 요구사항 7.1, 7.2, 7.3, 7.5**

- [x] 16. QuantStats 리포트 생성 모듈 구현 (`results/reporter.py`)
  - [x] 16.1 ReportGenerator 클래스 구현
    - 상위 N개(기본값: 5) 조합에 대해 QuantStats HTML 리포트 생성
    - S&P 500 Buy & Hold 수익률을 Benchmark로 비교 분석 포함
    - 파일명 형식: `rank{순위:02d}_{리딩지표}_{확인지표들}.html`
    - 핵심 지표(총 수익률, CAGR, Max Drawdown, Sharpe, Sortino) 콘솔 출력
    - 리포트 생성 중 오류 시 로그 기록 후 다음 조합 계속 생성
    - _요구사항: 8.1, 8.2, 8.3, 8.4, 8.5_
  - [ ]\* 16.2 리포트 파일명 형식 속성 테스트 작성
    - **Property 15: 리포트 파일명 형식**
    - **검증: 요구사항 8.3**

- [x] 17. 체크포인트 - 결과 및 리포트 모듈 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의

- [x] 18. 메인 오케스트레이터 및 전체 파이프라인 연결 (`main.py`)
  - [x] 18.1 메인 오케스트레이터 구현
    - config.yaml 로딩 → DataFetcher → CombinationEngine → ParameterOptimizer → SignalGenerator → BacktestEngine → ResultRanker → ReportGenerator 순서로 전체 파이프라인 연결
    - IndicatorCache를 파이프라인 전체에서 공유하여 중복 계산 방지
    - 멀티프로세싱 기반 병렬 처리 통합
    - 체크포인트 저장/복원 통합
    - 진행률 및 ETA 콘솔 출력
    - 명령줄 인자 없이 기본값으로 실행 가능
    - _요구사항: 9.1, 9.2, 9.5, 10.1, 10.2, 10.3_
  - [ ]\* 18.2 통합 테스트 작성
    - 소규모 데이터(지표 2~3개, 확인 지표 1~2개)로 전체 파이프라인 end-to-end 자동화 테스트
    - CSV 결과 파일 생성 검증
    - QuantStats HTML 리포트 생성 검증
    - _요구사항: 9.2_

- [x] 19. 최종 체크포인트 - 전체 시스템 검증
  - 모든 테스트 통과 확인, 질문이 있으면 사용자에게 문의

## 참고사항

- `*` 표시된 태스크는 선택사항이며 빠른 MVP를 위해 건너뛸 수 있음
- 각 태스크는 특정 요구사항을 참조하여 추적 가능
- 체크포인트에서 점진적 검증 수행
- 속성 테스트(Property-Based Test)는 Hypothesis 라이브러리를 사용하여 보편적 정확성 속성을 검증
- 단위 테스트는 특정 예시와 엣지 케이스를 검증
- 모든 지표 구현은 pandas/numpy 벡터화 연산을 사용하여 성능 최적화
