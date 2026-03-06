# Confirmation Indicator Pine Script 불일치 수정 설계

## Overview

`sp500_backtest/indicators/confirmation.py`에 구현된 14개 confirmation indicator가 원본 Pine Script(`strategy.py`)와 불일치하는 버그를 수정한다. Pine Script가 ground truth이며, Python 구현이 이를 정확히 반영해야 한다. 버그 범위는 기본값 오류(MODERATE)부터 공식/로직 완전 불일치(CRITICAL)까지 다양하며, 모든 수정은 단일 파일 `sp500_backtest/indicators/confirmation.py`에서 이루어진다.

## Glossary

- **Bug_Condition (C)**: 14개 confirmation indicator 중 하나가 Pine Script와 다른 시그널을 생성하는 조건
- **Property (P)**: 수정된 indicator가 Pine Script와 동일한 시그널을 생성하는 것
- **Preservation**: 수정 대상이 아닌 30개+ indicator와 수정 대상 indicator의 영향받지 않는 서브타입이 기존과 동일하게 동작하는 것
- **confirmation.py**: 45개+ confirmation indicator가 구현된 Python 파일 (3219줄)
- **strategy.py**: Pine Script v6 원본 코드 (ground truth)
- **ConfirmationIndicator**: `base.py`의 추상 클래스, `_calculate_impl()` 메서드를 통해 `IndicatorResult(long_signal, short_signal)` 반환
- **IndicatorResult**: `long_signal`과 `short_signal` boolean Series를 담는 데이터 클래스

## Bug Details

### Fault Condition

14개 confirmation indicator가 Pine Script와 다른 시그널을 생성한다. 버그는 기본값 오류, 비교 연산자 오류, 공식/로직 완전 불일치 등 3가지 카테고리로 분류된다.

**Formal Specification:**

```
FUNCTION isBugCondition(indicator_name, params, subtype)
  INPUT: indicator_name (str), params (dict), subtype (str | None)
  OUTPUT: boolean

  buggy_indicators = {
    "RSILimitConfirmation",
    "RSIMALimitConfirmation",
    "BullBearPowerTrendConfirmation",
    "CCIConfirmation",
    "IchimokuCloudConfirmation",
    "SuperIchiConfirmation",
    "TSIConfirmation" WHERE subtype == "Zero line cross",
    "BXtrenderConfirmation",
    "WaddahAttarExplosionConfirmation",
    "DonchianTrendRibbonConfirmation",
    "DMIADXConfirmation",
    "ChoppinessIndexConfirmation",
    "RSIMADirectionConfirmation",
    "RSIConfirmation" WHERE subtype == "RSI Level"
  }

  RETURN (indicator_name, subtype) IN buggy_indicators
END FUNCTION
```

### Examples

**🔴 CRITICAL — RSILimitConfirmation (기본값 + 로직 모두 틀림)**

- Python: `upper=70, lower=30`, `rsi < 70` → Long, `rsi > 30` → Short (거의 항상 True)
- Pine Script: `rsilimitup=40, rsilimitdown=60`, `rsi >= 40` → Long, `rsi <= 60` → Short
- 결과: RSI=50일 때 Python은 Long=True/Short=True, Pine Script는 Long=True/Short=True (우연히 일치하지만 RSI=35일 때 Python Long=True, Pine Script Long=False)

**🟠 MAJOR — BullBearPowerTrendConfirmation (공식 완전 불일치)**

- Python: `bull = high - EMA(close, 50)`, `bear = low - EMA(close, 50)` → 전통적 Elder 공식
- Pine Script: `BullTrend = (close - lowest(low, 50)) / ATR(5)`, `BearTrend = (highest(high, 50) - close) / ATR(5)` → 커스텀 정규화 공식 + 히스토그램 로직
- 결과: 완전히 다른 시그널 생성

**🟠 MAJOR — IchimokuCloudConfirmation (조건 수 불일치)**

- Python: `close > cloud_top` 단일 조건
- Pine Script: 5개 조건 동시 충족 (전환선>기준선, 선행스팬A>B, close>선행스팬[displacement-1], 치코스팬>선행스팬[50])
- 결과: Python이 훨씬 더 많은 Long 시그널 생성

**🟡 MODERATE — RSIMADirectionConfirmation (비교 연산자 오류)**

- Python: `rsi_ma > rsi_ma.shift(1)` (strict `>`)
- Pine Script: `rsiMA >= rsiMA[1]` (includes equal `>=`)
- 결과: RSI MA가 동일할 때 Python은 False, Pine Script는 True

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**

- 수정 대상이 아닌 30개+ confirmation indicator (EMAFilter, TwoEMACross, ThreeEMACross, RangeFilter, RQK, Supertrend, HalfTrend, ROC, McGinleyDynamic, DPO, BBOscillator, Stochastic, MACD, TrendlineBreakout, RangeDetector, HACOLT, ChandelierExit, ParabolicSAR, SSLChannel, HullSuite, AwesomeOscillator, VolatilityOscillator, DamianiVolatility, Volume, WolfpackId, QQEMod, ChaikinMoneyFlow, VortexIndicator, STC, VWAP)는 기존과 동일한 시그널 생성
- RSIConfirmation의 "RSI MA Cross", "RSI Exits OB-OS" 서브타입은 기존과 동일
- TSIConfirmation의 "Signal Cross" 서브타입은 기존과 동일
- DMIADXConfirmation의 "Adx Only", "Adx & +Di -Di" 서브타입은 기본값 변경 외 동일 로직 유지
- `_resolve_params` 메서드를 통한 사용자 파라미터 병합 동작 유지
- 유효하지 않은 서브타입 전달 시 `ValueError` 발생 유지
- 모든 결과는 `IndicatorResult(long_signal, short_signal)` 형식, NaN은 False로 채움

**Scope:**
수정 대상 14개 indicator의 `_calculate_impl` 메서드와 `default_params` 프로퍼티만 변경한다. 클래스 구조, 상속 관계, `name` 프로퍼티, `subtypes` 프로퍼티는 변경하지 않는다 (BXtrender의 subtypes 이름 변경 제외).

## Hypothesized Root Cause

14개 indicator 각각의 근본 원인을 분석한다:

1. **RSILimitConfirmation / RSIMALimitConfirmation**: 기본값이 전통적 RSI 과매수/과매도 레벨(70/30)로 설정됨. Pine Script는 커스텀 필터 레벨(40/60)을 사용. 비교 연산자도 `<`/`>` 대신 `>=`/`<=` 사용해야 함.

2. **BullBearPowerTrendConfirmation**: 전통적 Elder Bull/Bear Power 공식을 사용했으나, Pine Script는 완전히 다른 커스텀 공식 사용. `(close - lowest(low, 50)) / ATR(5)` 기반 정규화 + 히스토그램 로직이 필요.

3. **CCIConfirmation**: 제로라인(0) 기준 비교를 사용했으나, Pine Script는 상한/하한 밴드(100/-100) 기준 비교 사용.

4. **IchimokuCloudConfirmation**: 단순 `close > cloud_top` 조건만 구현했으나, Pine Script는 5개 조건(전환선>기준선, 선행스팬A>B, close>선행스팬[displacement-1] 2개, 치코스팬>선행스팬[50] 2개) 동시 충족 필요.

5. **SuperIchiConfirmation**: 표준 이치모쿠 donchian 계산을 사용했으나, Pine Script는 ATR 기반 trailing stop `avg()` 함수와 multiplier 파라미터를 사용하는 완전히 다른 커스텀 계산.

6. **TSIConfirmation (Zero line cross)**: `tsi > 0` 단일 조건만 확인했으나, Pine Script는 `tsi > signal AND tsi > 0` 두 조건 동시 요구.

7. **BXtrenderConfirmation**: `stoch_func` + SMA 기반 계산과 레벨 비교(`> 0`)를 사용했으나, Pine Script는 `ta.rsi(ema_diff, short_l3)` + T3 이동평균 기반 계산과 방향 비교(`> [1]`) 사용. `long_l1`, `long_l2` 파라미터도 누락.

8. **WaddahAttarExplosionConfirmation**: deadzone 필터가 누락됨. Pine Script는 `RMA(TR, 100) * 3.7` 기반 deadzone을 필수로 적용.

9. **DonchianTrendRibbonConfirmation**: 5개 기간 Donchian 중간값 합산 방식을 사용했으나, Pine Script는 단일 `dchannel(dlen)` 함수로 `close > highest[1]` 브레이크아웃 기반 로직 사용.

10. **DMIADXConfirmation**: 기본값 `length=14, adx_smoothing=14`를 사용했으나, Pine Script는 `dilen=10, adxlen=5`. Advance 서브타입의 adxcycle 및 DI 차이 > 1 조건도 누락.

11. **ChoppinessIndexConfirmation**: `trending_threshold=38.2`와 별도 `choppy_threshold=61.8`을 사용했으나, Pine Script는 단일 `ci_limit=61.8` 파라미터로 `ci < ci_limit` 조건만 사용.

12. **RSIMADirectionConfirmation**: `>` 연산자를 사용했으나, Pine Script는 `>=` 사용.

13. **RSIConfirmation (RSI Level)**: 레벨을 50으로 하드코딩했으나, Pine Script는 `respectrsilevel` 파라미터(기본값 50)로 설정 가능.

## Correctness Properties

Property 1: Fault Condition - RSILimitConfirmation 기본값 및 로직 수정

_For any_ OHLCV 입력에 대해 RSILimitConfirmation이 기본 파라미터로 실행될 때, 수정된 함수는 SHALL `upper=40, lower=60`을 기본값으로 사용하고 `rsi >= upper` → Long, `rsi <= lower` → Short 로직을 적용하여 Pine Script와 동일한 시그널을 생성한다.

**Validates: Requirements 2.1**

Property 2: Fault Condition - RSIMALimitConfirmation 기본값 및 로직 수정

_For any_ OHLCV 입력에 대해 RSIMALimitConfirmation이 기본 파라미터로 실행될 때, 수정된 함수는 SHALL `upper=40, lower=60`을 기본값으로 사용하고 `rsi_ma >= upper` → Long, `rsi_ma <= lower` → Short 로직을 적용하여 Pine Script와 동일한 시그널을 생성한다.

**Validates: Requirements 2.2**

Property 3: Fault Condition - BullBearPowerTrendConfirmation 공식 수정

_For any_ OHLCV 입력에 대해 BullBearPowerTrendConfirmation이 실행될 때, 수정된 함수는 SHALL Pine Script와 동일하게 `BullTrend = (close - lowest(low, 50)) / ATR(5)`, `BearTrend = (highest(high, 50) - close) / ATR(5)` 공식과 히스토그램 로직을 사용하여 시그널을 생성한다.

**Validates: Requirements 2.3**

Property 4: Fault Condition - CCIConfirmation 밴드 기반 비교 수정

_For any_ OHLCV 입력에 대해 CCIConfirmation이 실행될 때, 수정된 함수는 SHALL `cci > upper_band(100)` → Long, `cci < lower_band(-100)` → Short 로직을 적용한다.

**Validates: Requirements 2.4**

Property 5: Fault Condition - IchimokuCloudConfirmation 5개 조건 수정

_For any_ OHLCV 입력에 대해 IchimokuCloudConfirmation이 실행될 때, 수정된 함수는 SHALL 5개 조건(전환선>기준선, 선행스팬A>B, close>선행스팬A[displacement-1], close>선행스팬B[displacement-1], 치코스팬>선행스팬[50])을 동시에 확인한다.

**Validates: Requirements 2.5**

Property 6: Fault Condition - SuperIchiConfirmation ATR 기반 계산 수정

_For any_ OHLCV 입력에 대해 SuperIchiConfirmation이 실행될 때, 수정된 함수는 SHALL ATR 기반 trailing stop `avg()` 함수와 `tenkan_mult=2, kijun_mult=4, spanB_mult=6` 파라미터를 사용하여 Pine Script와 동일한 시그널을 생성한다.

**Validates: Requirements 2.6**

Property 7: Fault Condition - TSIConfirmation Zero line cross 수정

_For any_ OHLCV 입력에 대해 TSIConfirmation이 "Zero line cross" 서브타입으로 실행될 때, 수정된 함수는 SHALL `tsi > signal AND tsi > 0` 두 조건을 동시에 확인하여 Long 시그널을 생성한다.

**Validates: Requirements 2.7**

Property 8: Fault Condition - BXtrenderConfirmation RSI+T3 기반 계산 수정

_For any_ OHLCV 입력에 대해 BXtrenderConfirmation이 실행될 때, 수정된 함수는 SHALL `ta.rsi(ema_diff, short_l3)` + T3 이동평균 기반 계산과 방향 비교(`> [1]`)를 사용하고, `long_l1=5, long_l2=10` 파라미터를 포함한다.

**Validates: Requirements 2.8**

Property 9: Fault Condition - WaddahAttarExplosionConfirmation deadzone 추가

_For any_ OHLCV 입력에 대해 WaddahAttarExplosionConfirmation이 실행될 때, 수정된 함수는 SHALL deadzone(`RMA(TR, 100) * 3.7`) 필터를 적용하고 `trendUp > e1 AND e1 > deadzone AND trendUp > deadzone` 조건을 사용한다.

**Validates: Requirements 2.9**

Property 10: Fault Condition - DonchianTrendRibbonConfirmation 브레이크아웃 로직 수정

_For any_ OHLCV 입력에 대해 DonchianTrendRibbonConfirmation이 실행될 때, 수정된 함수는 SHALL `close > highest[1]` → trend=1, `close < lowest[1]` → trend=-1 브레이크아웃 기반 로직을 적용한다.

**Validates: Requirements 2.10**

Property 11: Fault Condition - DMIADXConfirmation 기본값 및 Advance 서브타입 수정

_For any_ OHLCV 입력에 대해 DMIADXConfirmation이 실행될 때, 수정된 함수는 SHALL `length=10, adx_smoothing=5`를 기본값으로 사용하고, Advance 서브타입에서 adxcycle 및 DI 차이 > 1 조건을 포함한다.

**Validates: Requirements 2.11**

Property 12: Fault Condition - ChoppinessIndexConfirmation 단일 임계값 수정

_For any_ OHLCV 입력에 대해 ChoppinessIndexConfirmation이 실행될 때, 수정된 함수는 SHALL 단일 `ci_limit=61.8` 파라미터로 `ci < ci_limit` 조건을 적용한다.

**Validates: Requirements 2.12**

Property 13: Fault Condition - RSIMADirectionConfirmation 비교 연산자 수정

_For any_ OHLCV 입력에 대해 RSIMADirectionConfirmation이 실행될 때, 수정된 함수는 SHALL `rsi_ma >= rsi_ma.shift(1)` → Long, `rsi_ma <= rsi_ma.shift(1)` → Short 비교를 사용한다.

**Validates: Requirements 2.13**

Property 14: Fault Condition - RSIConfirmation RSI Level 파라미터화

_For any_ OHLCV 입력에 대해 RSIConfirmation이 "RSI Level" 서브타입으로 실행될 때, 수정된 함수는 SHALL `level` 파라미터(기본값 50)를 사용하여 `rsi > level` → Long, `rsi < level` → Short 로직을 적용한다.

**Validates: Requirements 2.14**

Property 15: Preservation - 수정 대상 외 indicator 동작 보존

_For any_ OHLCV 입력에 대해 수정 대상이 아닌 confirmation indicator가 실행될 때, 수정된 코드는 SHALL 기존과 동일한 시그널을 생성하여 모든 비수정 indicator의 동작을 보존한다.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

## Fix Implementation

### Changes Required

근본 원인 분석이 정확하다는 가정 하에, 모든 변경은 `sp500_backtest/indicators/confirmation.py` 단일 파일에서 이루어진다.

**File**: `sp500_backtest/indicators/confirmation.py`

**1. RSILimitConfirmation (Line ~1353)**

- `default_params`: `upper=70` → `upper=40`, `lower=30` → `lower=60`
- `_calculate_impl`: `rsi_val < upper` → `rsi_val >= upper`, `rsi_val > lower` → `rsi_val <= lower`

**2. RSIMALimitConfirmation (Line ~1407)**

- `default_params`: `upper=70` → `upper=40`, `lower=30` → `lower=60`
- `_calculate_impl`: `rsi_ma < upper` → `rsi_ma >= upper`, `rsi_ma > lower` → `rsi_ma <= lower`

**3. BullBearPowerTrendConfirmation (Line ~950)**

- `_calculate_impl` 전체 재작성:
  - `BullTrend = (close - lowest(low, 50)) / ATR(5)`
  - `BearTrend = (highest(high, 50) - close) / ATR(5)`
  - `BearTrend2 = -1 * BearTrend`
  - `Trend = BullTrend - BearTrend`
  - 히스토그램 로직: `BullTrend < 2` → `BullTrend_hist = BullTrend - 2`, `BearTrend2 > -2` → `BearTrend_hist = BearTrend2 + 2`
  - Follow Trend: `BearTrend_hist > 0 AND Trend >= 2` → Long
  - Without Trend: `BearTrend_hist > 0` → Long

**4. CCIConfirmation (Line ~1967)**

- `default_params`에 `upper_band=100`, `lower_band=-100` 추가
- `_calculate_impl`: `cci > 0` → `cci > upper_band`, `cci < 0` → `cci < lower_band`

**5. IchimokuCloudConfirmation (Line ~1544)**

- `default_params`에 `displacement=26` 추가
- `_calculate_impl` 전체 재작성: 5개 조건 동시 충족
  - `conversionLine > baseLine`
  - `leadLine1 > leadLine2`
  - `close > leadLine1.shift(displacement - 1)`
  - `close > leadLine2.shift(displacement - 1)`
  - `ChikouSpan > leadLine1.shift(50) AND ChikouSpan > leadLine2.shift(50)`
  - 여기서 `ChikouSpan = close` (Pine Script의 `close[25] + (close - close[25])` = `close`)

**6. SuperIchiConfirmation (Line ~1619)**

- `default_params` 전체 재작성: `tenkan_len=9, tenkan_mult=2.0, kijun_len=26, kijun_mult=4.0, spanB_len=52, spanB_mult=6.0, displacement=26`
- ATR 기반 trailing stop `avg()` 함수 구현 (Pine Script의 `avg(src, length, mult)` 로직)
- 6개 조건 동시 충족: `tenkan > kijun AND senkouA > senkouB AND close > senkouA[displacement-1] AND close > senkouB[displacement-1] AND ChikouSpan > senkouA[50] AND ChikouSpan > senkouB[50]`

**7. TSIConfirmation (Line ~796)**

- `_calculate_impl`의 "Zero line cross" 분기: `tsi > 0` → `(tsi > signal_line) & (tsi > 0)`, `tsi < 0` → `(tsi < signal_line) & (tsi < 0)`

**8. BXtrenderConfirmation (Line ~872)**

- `default_params`에 `long_l1=5, long_l2=10` 추가
- `_calculate_impl` 전체 재작성:
  - `shortTermXtrender = rsi(ema(close, short_l1) - ema(close, short_l2), short_l3) - 50`
  - `maShortTermXtrender = t3(shortTermXtrender, 5)` (T3 이동평균 구현 필요)
  - Short term: `maShortTermXtrender > maShortTermXtrender[1]` → Long
  - Short and Long term: 위 조건 + `longTermXtrender > 0 AND longTermXtrender > longTermXtrender[1] AND shortTermXtrender > shortTermXtrender[1] AND shortTermXtrender > 0`
  - `longTermXtrender = rsi(ema(close, long_l1), long_l2) - 50`
- `subtypes` 이름 변경: `"Short term"` → `"Short Term trend"`, `"Short and Long term"` → `"Short and Long term trend"`

**9. WaddahAttarExplosionConfirmation (Line ~2423)**

- `_calculate_impl`에 deadzone 계산 추가: `deadzone = RMA(TR, 100) * 3.7`
- `trendUp = max(trend, 0)`, `trendDown = max(-trend, 0)` 분리
- Long 조건: `trendUp > 0 AND trendUp > e1 AND e1 > deadzone AND trendUp > deadzone`
- Short 조건: `trendDown > 0 AND trendDown > e1 AND e1 > deadzone AND trendDown > deadzone`

**10. DonchianTrendRibbonConfirmation (Line ~554)**

- `_calculate_impl` 전체 재작성:
  - `hh = highest(high, period)`, `ll = lowest(low, period)`
  - `trend = 1 if close > hh[1] else (-1 if close < ll[1] else trend[t-1])`
  - 상태 기반 루프로 구현 (이전 trend 값 유지)

**11. DMIADXConfirmation (Line ~2332)**

- `default_params`: `length=14` → `length=10`, `adx_smoothing=14` → `adx_smoothing=5`
- Advance 서브타입 전체 재작성:
  - `adxcycle` 상태 변수 추가: `adx crossover keyLevel` → 1, `adx crossunder keyLevel` → -1
  - `adxcycle == -1`: `diplus > diminus AND adx >= keyLevel AND diplus - diminus > 1`
  - `adxcycle == 1`: 위 조건 + `adx < 55 AND (adx > adx[1] OR (diplus > diplus[1] AND diminus < diminus[1]))`

**12. ChoppinessIndexConfirmation (Line ~2550)**

- `default_params`: `trending_threshold=38.2, choppy_threshold=61.8` → `ci_limit=61.8`
- `_calculate_impl`: `ci < trending` → `ci < ci_limit`

**13. RSIMADirectionConfirmation (Line ~1300)**

- `_calculate_impl`: `rsi_ma > rsi_ma.shift(1)` → `rsi_ma >= rsi_ma.shift(1)`, `rsi_ma < rsi_ma.shift(1)` → `rsi_ma <= rsi_ma.shift(1)`

**14. RSIConfirmation (Line ~1202)**

- `default_params`에 `level=50` 추가
- `_calculate_impl`의 "RSI Level" 분기: 하드코딩 `50` → `params["level"]` 사용

## Testing Strategy

### Validation Approach

테스트 전략은 2단계로 진행한다: (1) 수정 전 코드에서 버그를 재현하는 탐색적 테스트, (2) 수정 후 코드에서 올바른 동작을 검증하는 수정 확인 + 보존 확인 테스트.

### Exploratory Fault Condition Checking

**Goal**: 수정 전 코드에서 14개 indicator의 버그를 재현하여 근본 원인 분석을 확인/반박한다.

**Test Plan**: 각 indicator에 대해 알려진 OHLCV 데이터를 입력하고, Python 출력과 Pine Script 기대값을 비교한다. 수정 전 코드에서 실패를 관찰한다.

**Test Cases**:

1. **RSILimit 기본값 테스트**: RSI=35인 데이터에서 `RSILimitConfirmation()`의 Long 시그널 확인 (수정 전: True, Pine Script 기대: False → 실패)
2. **BullBearPower 공식 테스트**: 알려진 OHLCV에서 `BullBearPowerTrendConfirmation()`의 시그널이 Pine Script와 다름을 확인 (수정 전: 실패)
3. **Ichimoku 조건 수 테스트**: 전환선<기준선이지만 close>cloud_top인 데이터에서 Long 시그널 확인 (수정 전: True, Pine Script 기대: False → 실패)
4. **TSI Zero line 테스트**: tsi>0이지만 tsi<signal인 데이터에서 Long 시그널 확인 (수정 전: True, Pine Script 기대: False → 실패)

**Expected Counterexamples**:

- 기본값 오류: RSI=35일 때 RSILimit Long이 True (Pine Script에서는 False)
- 로직 오류: 단일 조건만 충족하는 데이터에서 Ichimoku Long이 True (Pine Script에서는 False)
- 공식 오류: BullBearPower가 완전히 다른 값 생성

### Fix Checking

**Goal**: 수정된 14개 indicator가 Pine Script와 동일한 시그널을 생성하는지 검증한다.

**Pseudocode:**

```
FOR ALL (indicator, params, subtype, df) WHERE isBugCondition(indicator, params, subtype) DO
  result := indicator_fixed.calculate(df, params, subtype)
  expected := pine_script_reference_output(indicator, params, subtype, df)
  ASSERT result.long_signal == expected.long_signal
  ASSERT result.short_signal == expected.short_signal
END FOR
```

### Preservation Checking

**Goal**: 수정 대상이 아닌 indicator와 수정 대상 indicator의 영향받지 않는 서브타입이 기존과 동일하게 동작하는지 검증한다.

**Pseudocode:**

```
FOR ALL (indicator, params, subtype, df) WHERE NOT isBugCondition(indicator, params, subtype) DO
  ASSERT indicator_original.calculate(df, params, subtype) == indicator_fixed.calculate(df, params, subtype)
END FOR
```

**Testing Approach**: Property-based testing을 사용하여 다양한 OHLCV 데이터에 대해 보존 검증을 수행한다. 랜덤 OHLCV DataFrame을 생성하고, 수정 전/후 코드의 출력을 비교한다.

**Test Plan**: 수정 전 코드에서 비수정 indicator의 동작을 관찰한 후, 수정 후 코드에서 동일한 동작을 검증한다.

**Test Cases**:

1. **비수정 Indicator 보존**: EMAFilter, Supertrend, MACD 등 30개+ indicator가 수정 전/후 동일한 시그널 생성
2. **RSI MA Cross 보존**: RSIConfirmation의 "RSI MA Cross" 서브타입이 수정 전/후 동일
3. **TSI Signal Cross 보존**: TSIConfirmation의 "Signal Cross" 서브타입이 수정 전/후 동일
4. **DMI Adx Only 보존**: DMIADXConfirmation의 "Adx Only" 서브타입이 기본값 변경 외 동일 로직

### Unit Tests

- 각 수정 indicator에 대해 알려진 OHLCV 데이터로 기대 시그널 검증
- 기본 파라미터 값이 Pine Script와 일치하는지 검증
- 비교 연산자(`>=` vs `>`)가 올바른지 경계값 테스트
- 새로 추가된 파라미터(deadzone, ci_limit, level 등)가 올바르게 동작하는지 검증
- SuperIchi의 ATR 기반 `avg()` 함수가 올바르게 구현되었는지 검증
- BXtrender의 T3 이동평균이 올바르게 구현되었는지 검증
- DonchianTrendRibbon의 상태 기반 브레이크아웃 로직이 올바르게 구현되었는지 검증
- DMI Advance 서브타입의 adxcycle 상태 머신이 올바르게 동작하는지 검증

### Property-Based Tests

- 랜덤 OHLCV DataFrame 생성 후 수정된 indicator의 출력이 `IndicatorResult` 형식인지 검증
- 랜덤 OHLCV DataFrame 생성 후 비수정 indicator의 수정 전/후 출력 동일성 검증
- RSILimit/RSIMALimit: 랜덤 RSI 값에 대해 `>=`/`<=` 비교가 올바른지 검증
- ChoppinessIndex: 랜덤 CI 값에 대해 단일 `ci_limit` 임계값 동작 검증
- WaddahAttar: 랜덤 OHLCV에서 deadzone 필터가 항상 적용되는지 검증

### Integration Tests

- 전체 백테스트 파이프라인에서 수정된 indicator를 사용하여 시그널 생성 검증
- 여러 confirmation indicator를 동시에 활성화한 상태에서 최종 시그널 조합 검증
- 수정된 indicator의 파라미터를 사용자 지정값으로 변경했을 때 올바르게 동작하는지 검증
