# 요구사항 문서

## 소개

TradingView Pine Script v6 기반 "DIY Custom Strategy Builder [ZP]"의 37개 리딩 지표와 45개 이상의 확인 지표를 Python으로 재구현하고, 최근 3년간 S&P 500 지수를 대상으로 모든 가능한 리딩+확인 지표 조합 및 파라미터 수치를 조절하여 Long/Short 백테스팅을 수행한다. 모든 조합의 수익률을 계산하여 높은 순으로 정렬하고, QuantStats를 통해 상위 조합의 성과 분석 리포트를 생성한다.

## 용어집

- **Backtest_Engine**: S&P 500 데이터를 기반으로 지표 조합별 매매 시그널을 생성하고 포트폴리오 수익률을 계산하는 Python 백테스팅 시스템
- **Data_Fetcher**: yfinance를 사용하여 S&P 500 가격 데이터를 수집하는 모듈
- **Indicator_Library**: Pine Script의 37개 리딩 지표와 45개 이상의 확인 지표를 Python으로 재구현한 지표 라이브러리
- **Signal_Generator**: 리딩 지표 1개와 확인 지표 0~N개의 조합으로 Long/Short 시그널을 생성하는 모듈
- **Combination_Engine**: 모든 가능한 리딩+확인 지표 조합과 파라미터 수치 변형을 생성하고 관리하는 모듈
- **Parameter_Optimizer**: 각 지표의 수치 파라미터(기간, 임계값, 승수 등)를 조절하여 최적 조합을 탐색하는 모듈
- **Result_Ranker**: 모든 조합의 백테스팅 결과를 수익률 기준으로 정렬하고 순위를 매기는 모듈
- **Report_Generator**: QuantStats를 사용하여 상위 조합의 성과 분석 리포트를 생성하는 모듈
- **Leading_Indicator**: 매매 시그널을 최초 발생시키는 주요 지표 (37개 옵션)
- **Confirmation_Indicator**: 리딩 지표의 시그널을 필터링하여 확인하는 보조 지표 (45개 이상 옵션)
- **Benchmark**: 성과 비교 기준이 되는 S&P 500 Buy & Hold 수익률
- **Strategy_Return**: 매매 시그널에 따라 Long/Short 포지션을 변경하여 얻은 전략 수익률

## 요구사항

### 요구사항 1: S&P 500 가격 데이터 수집

**사용자 스토리:** 트레이더로서, 최근 3년간 S&P 500 일봉 데이터를 자동으로 수집하고 싶다. 이를 통해 백테스팅에 필요한 OHLCV 데이터를 확보할 수 있다.

#### 인수 조건 (데이터 수집)

1. WHEN 백테스팅 스크립트가 실행되면, THE Data_Fetcher SHALL yfinance를 사용하여 최근 3년간 S&P 500(^GSPC) 일봉 OHLCV 데이터를 수집한다
2. WHEN 데이터 수집이 완료되면, THE Data_Fetcher SHALL 수집된 데이터의 시작일, 종료일, 총 거래일 수를 콘솔에 출력한다
3. IF 네트워크 오류 또는 데이터 소스 장애가 발생하면, THEN THE Data_Fetcher SHALL 오류 메시지를 출력하고 프로그램을 안전하게 종료한다
4. THE Data_Fetcher SHALL 수집된 데이터에서 결측값(NaN)을 전방 채움(forward fill) 방식으로 처리한다
5. THE Data_Fetcher SHALL 거래량(Volume) 데이터를 포함하여 수집한다 (Volume 기반 지표 계산에 필요)

### 요구사항 2: Pine Script 리딩 지표 Python 재구현

**사용자 스토리:** 트레이더로서, Pine Script에 정의된 37개 리딩 지표를 Python으로 동일하게 재구현하고 싶다. 이를 통해 TradingView 전략과 동일한 시그널을 Python 환경에서 생성할 수 있다.

#### 인수 조건 (리딩 지표 구현)

1. THE Indicator_Library SHALL 다음 37개 리딩 지표를 각각 독립적인 함수로 구현한다: Range Filter, Rational Quadratic Kernel (RQK), Supertrend, Half Trend, Ichimoku Cloud, SuperIchi, True Strength Indicator (TSI), Trend Direction Force Index (TDFI), Trendline Breakout, Range Detector, Heiken-Ashi Candlestick Oscillator, Donchian Trend Ribbon, Stochastic, RSI, Rate of Change (ROC), VWAP, CCI, 2 EMA Cross, 3 EMA Cross, B-Xtrender, Bull Bear Power Trend, Detrended Price Oscillator (DPO), BB Oscillator, Chandelier Exit, DMI (ADX), Parabolic SAR (PSAR), MACD, SSL Channel, Waddah Attar Explosion, Chaikin Money Flow, Vortex Index, Schaff Trend Cycle (STC), Awesome Oscillator, Volatility Oscillator, Wolfpack Id, QQE Mod, Hull Suite
2. WHEN 리딩 지표 함수가 호출되면, THE Indicator_Library SHALL 해당 지표의 Long 시그널(True/False)과 Short 시그널(True/False)을 pandas Series로 반환한다
3. THE Indicator_Library SHALL 각 리딩 지표의 기본 파라미터를 Pine Script의 기본값과 동일하게 설정한다
4. THE Indicator_Library SHALL 각 지표 함수에 파라미터를 딕셔너리로 전달할 수 있는 인터페이스를 제공한다

### 요구사항 3: Pine Script 확인 지표 Python 재구현

**사용자 스토리:** 트레이더로서, Pine Script에 정의된 45개 이상의 확인 지표를 Python으로 재구현하고 싶다. 이를 통해 리딩 지표의 시그널을 필터링하여 정확도를 높일 수 있다.

#### 인수 조건 (확인 지표 구현)

1. THE Indicator_Library SHALL 다음 확인 지표를 각각 독립적인 함수로 구현한다: EMA Filter, 2 EMA Cross, 3 EMA Cross, Range Filter (Default/DW), Rational Quadratic Kernel (RQK), SuperTrend, Half Trend, Donchian Trend Ribbon, Rate of Change (ROC), True Strength Indicator (TSI, Signal Cross/Zero line cross), Trend Direction Force Index (TDFI), McGinley Dynamic, Detrended Price Oscillator (DPO), Ichimoku Cloud, SuperIchi, Trendline Breakout, Range Detector, Heiken-Ashi Candlestick Oscillator, B-Xtrender (Short term/Short and Long term), Bull Bear Power Trend (Follow Trend/Without Trend), VWAP, BB Oscillator (Entering/Exiting Band), Chandelier Exit, CCI, Awesome Oscillator (Zero Line Cross/AC Zero Line Cross/AC Momentum Bar), DMI-ADX (Adx Only/Adx & +Di -Di/Advance), Parabolic SAR, Waddah Attar Explosion, Volatility Oscillator, Choppiness Index, Damiani Volatility (Simple/Threshold/10p Difference), Stochastic (CrossOver/CrossOver in OB & OS levels/%K above-below %D), RSI (RSI MA Cross/RSI Exits OB-OS zones/RSI Level), RSI MA Direction, RSI Limit, RSI MA Limit, MACD (MACD Crossover/Zero line crossover), SSL Channel, Schaff Trend Cycle (STC), Chaikin Money Flow, Volume (volume above MA/Simple/Delta), Wolfpack Id, QQE Mod (Line/Bar/Line & Bar), Hull Suite, Vortex Indicator (Simple/Advance)
2. WHEN 확인 지표 함수가 호출되면, THE Indicator_Library SHALL 해당 지표의 Long 확인(True/False)과 Short 확인(True/False)을 pandas Series로 반환한다
3. THE Indicator_Library SHALL 서브타입이 있는 확인 지표(예: TSI의 Signal Cross/Zero line cross, Stochastic의 CrossOver/OB&OS levels 등)에 대해 서브타입 파라미터를 지원한다
4. THE Indicator_Library SHALL 각 확인 지표의 기본 파라미터를 Pine Script의 기본값과 동일하게 설정한다

### 요구사항 4: 지표 조합 생성 및 관리

**사용자 스토리:** 트레이더로서, 37개 리딩 지표와 45개 이상의 확인 지표의 모든 가능한 조합을 자동으로 생성하고 싶다. 이를 통해 최적의 지표 조합을 체계적으로 탐색할 수 있다.

#### 인수 조건 (조합 생성)

1. THE Combination_Engine SHALL 37개 리딩 지표 중 1개를 선택하고, 확인 지표 0개~최대 N개(설정 가능, 기본값: 3개)를 조합하여 모든 가능한 조합을 생성한다
2. THE Combination_Engine SHALL 서브타입이 있는 지표(예: TSI의 Signal Cross/Zero line cross)를 별도 조합으로 취급한다
3. THE Combination_Engine SHALL 생성된 총 조합 수를 콘솔에 출력한다
4. WHEN 조합 수가 설정된 최대 한도(기본값: 100,000)를 초과하면, THE Combination_Engine SHALL 확인 지표 최대 개수를 자동으로 줄여 한도 이내로 조정한다
5. THE Combination_Engine SHALL 각 조합에 고유 식별자를 부여하여 결과 추적이 가능하도록 한다

### 요구사항 5: 파라미터 최적화

**사용자 스토리:** 트레이더로서, 각 지표의 수치 파라미터(기간, 임계값, 승수 등)를 조절하여 최적의 수치를 찾고 싶다. 이를 통해 단순 기본값 이상의 수익률을 달성할 수 있다.

#### 인수 조건 (파라미터 최적화)

1. THE Parameter_Optimizer SHALL 각 지표의 주요 수치 파라미터에 대해 탐색 범위를 정의한다 (예: EMA 기간 5~200, Supertrend ATR 기간 5~30, RSI 기간 7~21 등)
2. THE Parameter_Optimizer SHALL 그리드 서치(Grid Search) 또는 랜덤 서치(Random Search) 방식으로 파라미터 조합을 탐색한다
3. THE Parameter_Optimizer SHALL 파라미터 탐색 범위와 스텝 크기를 설정 파일 또는 딕셔너리로 정의할 수 있는 인터페이스를 제공한다
4. WHEN 파라미터 최적화가 실행되면, THE Parameter_Optimizer SHALL 현재 진행률(완료된 조합 수 / 전체 조합 수)을 주기적으로 콘솔에 출력한다
5. THE Parameter_Optimizer SHALL 멀티프로세싱을 활용하여 병렬로 백테스팅을 수행한다

### 요구사항 6: Long/Short 시그널 생성 및 백테스팅

**사용자 스토리:** 트레이더로서, 각 지표 조합에 대해 Long과 Short 양방향 시그널을 생성하고 백테스팅을 수행하고 싶다. 이를 통해 양방향 매매 전략의 수익성을 평가할 수 있다.

#### 인수 조건 (시그널 생성 및 백테스팅)

1. WHEN 리딩 지표가 Long 시그널을 발생시키고 모든 활성 확인 지표가 Long을 확인하면, THE Signal_Generator SHALL Long 포지션 진입 시그널을 생성한다
2. WHEN 리딩 지표가 Short 시그널을 발생시키고 모든 활성 확인 지표가 Short을 확인하면, THE Signal_Generator SHALL Short 포지션 진입 시그널을 생성한다
3. THE Signal_Generator SHALL Pine Script의 Signal Expiry 로직을 구현하여, 리딩 지표 시그널 발생 후 설정된 캔들 수(기본값: 3) 이내에 확인 지표가 확인되지 않으면 시그널을 무효화한다
4. THE Signal_Generator SHALL Alternate Signal 옵션을 구현하여, 연속 동일 방향 시그널을 필터링한다
5. THE Backtest_Engine SHALL Long 포지션을 +1, Short 포지션을 -1, 무포지션을 0으로 설정하여 수익률을 계산한다
6. THE Backtest_Engine SHALL 시그널 발생 다음 거래일부터 포지션을 반영한다 (미래 정보 편향 방지)
7. THE Backtest_Engine SHALL 거래 비용(기본값: 0.1%)을 포지션 변경 시 차감하는 옵션을 제공한다

### 요구사항 7: 결과 정렬 및 순위 산출

**사용자 스토리:** 트레이더로서, 모든 지표 조합의 백테스팅 결과를 수익률 높은 순으로 정렬하여 확인하고 싶다. 이를 통해 가장 수익성 높은 전략 조합을 빠르게 식별할 수 있다.

#### 인수 조건 (결과 정렬)

1. WHEN 모든 조합의 백테스팅이 완료되면, THE Result_Ranker SHALL 총 수익률(Total Return) 기준으로 내림차순 정렬한다
2. THE Result_Ranker SHALL 각 조합에 대해 다음 지표를 계산하여 결과 테이블에 포함한다: 총 수익률, 연환산 수익률(CAGR), 최대 낙폭(Max Drawdown), 샤프 비율(Sharpe Ratio), 총 거래 횟수, 승률(Win Rate)
3. THE Result_Ranker SHALL 결과를 CSV 파일로 저장한다
4. THE Result_Ranker SHALL 상위 N개(기본값: 20) 조합의 요약을 콘솔에 출력한다
5. THE Result_Ranker SHALL 정렬 기준을 총 수익률, CAGR, 샤프 비율 중 선택할 수 있는 옵션을 제공한다

### 요구사항 8: QuantStats 성과 리포트 생성

**사용자 스토리:** 트레이더로서, 상위 수익률 조합에 대해 QuantStats 기반의 상세 성과 리포트를 생성하고 싶다. 이를 통해 전략의 강점과 약점을 시각적으로 분석할 수 있다.

#### 인수 조건 (리포트 생성)

1. WHEN 결과 정렬이 완료되면, THE Report_Generator SHALL 상위 N개(기본값: 5) 조합에 대해 QuantStats HTML 리포트를 각각 생성한다
2. THE Report_Generator SHALL S&P 500 Buy & Hold 수익률을 Benchmark로 사용하여 비교 분석을 포함한다
3. THE Report_Generator SHALL 각 리포트 파일명에 순위와 조합 정보를 포함한다 (예: rank01_RangeFilter_RQK_SuperTrend.html)
4. THE Report_Generator SHALL 다음 핵심 지표를 콘솔에 출력한다: 총 수익률, 연환산 수익률(CAGR), 최대 낙폭(Max Drawdown), 샤프 비율(Sharpe Ratio), 소르티노 비율(Sortino Ratio)
5. IF 리포트 생성 중 오류가 발생하면, THEN THE Report_Generator SHALL 오류 내용을 로그에 기록하고 다음 조합의 리포트 생성을 계속한다

### 요구사항 9: 실행 설정 및 환경 구성

**사용자 스토리:** 트레이더로서, 백테스팅 파라미터와 실행 환경을 쉽게 설정하고 싶다. 이를 통해 다양한 조건에서 전략 탐색을 유연하게 수행할 수 있다.

#### 인수 조건 (실행 설정)

1. THE Backtest_Engine SHALL 설정 파일(config.yaml 또는 config.py)을 통해 다음 항목을 설정할 수 있도록 한다: 백테스팅 기간, 최대 확인 지표 수, 최대 조합 수 한도, 파라미터 탐색 범위, 거래 비용, 리포트 생성 개수, 정렬 기준, 병렬 처리 워커 수
2. THE Backtest_Engine SHALL 명령줄 인자 없이 기본값으로 실행 가능하도록 구성한다
3. THE Backtest_Engine SHALL Python 3.9 이상 환경에서 동작한다
4. THE Backtest_Engine SHALL 필요한 패키지 목록(yfinance, quantstats, pandas, numpy, ta-lib 또는 pandas-ta, pyyaml)을 requirements.txt 파일로 제공한다
5. THE Backtest_Engine SHALL 중간 결과를 주기적으로 저장하여, 중단 후 재시작 시 이어서 실행할 수 있는 체크포인트 기능을 제공한다

### 요구사항 10: 성능 및 확장성

**사용자 스토리:** 트레이더로서, 대량의 조합을 합리적인 시간 내에 처리하고 싶다. 이를 통해 실용적인 시간 내에 최적 전략을 탐색할 수 있다.

#### 인수 조건 (성능)

1. THE Backtest_Engine SHALL 멀티프로세싱(multiprocessing 또는 concurrent.futures)을 활용하여 CPU 코어 수에 비례하는 병렬 처리를 수행한다
2. THE Backtest_Engine SHALL 지표 계산 결과를 캐싱하여, 동일 지표가 여러 조합에서 사용될 때 중복 계산을 방지한다
3. WHEN 백테스팅이 진행 중일 때, THE Backtest_Engine SHALL 예상 완료 시간(ETA)과 현재 진행률을 콘솔에 출력한다
4. THE Backtest_Engine SHALL 벡터화 연산(pandas/numpy)을 사용하여 지표 계산과 수익률 계산을 수행한다
