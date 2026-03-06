# Bugfix Requirements Document

## Introduction

`sp500_backtest/indicators/confirmation.py`에 구현된 14개 confirmation indicator가 원본 Pine Script(`strategy.py`, Pine Script v6 "DIY Custom Strategy Builder [ZP]")와 불일치합니다. `strategy.py`가 ground truth(정답)이며, Python 구현이 이를 정확히 반영해야 합니다. 기본값 오류, 비교 연산자 오류, 공식/로직 완전 불일치 등 다양한 심각도의 버그가 포함되어 있습니다.

## Bug Analysis

### Current Behavior (Defect)

**🔴 CRITICAL — 기본값 + 로직 모두 틀림**

1.1 WHEN RSILimitConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 `upper=70, lower=30`을 사용하고 `rsi < upper` → Long, `rsi > lower` → Short 로직을 적용하여 Pine Script의 `rsilimitup=40, rsilimitdown=60`, `rsi >= rsilimitup` → Long 조건과 완전히 다른 시그널을 생성한다

1.2 WHEN RSIMALimitConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 `upper=70, lower=30`을 사용하고 `rsi_ma < upper` → Long, `rsi_ma > lower` → Short 로직을 적용하여 Pine Script의 `rsimalimitup=40, rsimalimitdown=60`, `rsiMA >= rsimalimitup` → Long 조건과 완전히 다른 시그널을 생성한다

**🟠 MAJOR — 로직/공식 틀림**

1.3 WHEN BullBearPowerTrendConfirmation이 실행될 때 THEN 시스템은 `bull = high - EMA(close, 50)`, `bear = low - EMA(close, 50)` 공식을 사용하지만, Pine Script는 `BullTrend = (close - lowest(low, 50)) / ATR(5)`, `BearTrend = (highest(high, 50) - close) / ATR(5)` 기반의 완전히 다른 공식과 히스토그램 로직을 사용한다

1.4 WHEN CCIConfirmation이 실행될 때 THEN 시스템은 `cci > 0` → Long, `cci < 0` → Short 로직을 사용하지만, Pine Script는 `cci > cciupperband(100)` → Long, `cci < ccilowerband(-100)` → Short로 상한/하한 밴드 기반 비교를 사용한다

1.5 WHEN IchimokuCloudConfirmation이 실행될 때 THEN 시스템은 `close > cloud_top` 단일 조건만 확인하지만, Pine Script는 전환선 > 기준선, 선행스팬A > 선행스팬B, close > 선행스팬(displacement 오프셋), 치코스팬 > 선행스팬(50봉 전) 등 5개 조건을 동시에 확인한다

1.6 WHEN SuperIchiConfirmation이 실행될 때 THEN 시스템은 표준 이치모쿠 donchian 계산을 사용하지만, Pine Script는 ATR 기반 trailing stop `avg()` 함수와 `tenkan_mult=2, kijun_mult=4, spanB_mult=6` 파라미터를 사용하는 완전히 다른 커스텀 계산을 사용한다

1.7 WHEN TSIConfirmation이 "Zero line cross" 서브타입으로 실행될 때 THEN 시스템은 `tsi > 0` 단일 조건만 확인하지만, Pine Script는 `tsi > signal AND tsi > 0` 두 조건을 동시에 요구한다

1.8 WHEN BXtrenderConfirmation이 실행될 때 THEN 시스템은 `stoch_func` + SMA 기반 계산과 레벨 기반 비교(`> 0` / `< 0`)를 사용하지만, Pine Script는 `ta.rsi(ema_diff, short_l3)` + T3 이동평균 기반 계산과 방향 기반 비교(`> [1]` / `< [1]`)를 사용하며, 장기 계산에 `long_l1=5, long_l2=10` 파라미터도 누락되어 있다

1.9 WHEN WaddahAttarExplosionConfirmation이 실행될 때 THEN 시스템은 `trend > 0 AND trend > explosion` → Long 조건만 확인하고 deadzone 필터가 없지만, Pine Script는 `trendUp > e1 AND e1 > deadzone AND trendUp > deadzone` 조건으로 deadzone(`RMA(TR, 100) * 3.7`) 필터를 필수로 적용한다

1.10 WHEN DonchianTrendRibbonConfirmation이 실행될 때 THEN 시스템은 5개 기간의 Donchian 채널 중간값 대비 종가 위치를 합산하는 trend_sum 방식을 사용하지만, Pine Script는 단일 `dchannel(dlen)` 함수로 `close > highest[1]` → trend=1, `close < lowest[1]` → trend=-1 브레이크아웃 기반 로직을 사용한다

**🟡 MODERATE — 기본값 틀림**

1.11 WHEN DMIADXConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 `length=14, adx_smoothing=14`를 사용하지만, Pine Script는 `dilen=10, adxlen=5`를 기본값으로 사용하며, Advance 서브타입의 adxcycle 및 DI 차이 > 1 조건도 누락되어 있다

1.12 WHEN ChoppinessIndexConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 `trending_threshold=38.2`와 별도의 `choppy_threshold=61.8`을 사용하지만, Pine Script는 단일 `ci_limit=61.8` 파라미터로 `ci < ci_limit` 조건만 사용한다

1.13 WHEN RSIMADirectionConfirmation이 실행될 때 THEN 시스템은 `rsi_ma > rsi_ma.shift(1)` (strict `>`) 비교를 사용하지만, Pine Script는 `rsiMA >= rsiMA[1]` (includes equal `>=`) 비교를 사용한다

**🔵 MINOR**

1.14 WHEN RSIConfirmation이 "RSI Level" 서브타입으로 실행될 때 THEN 시스템은 레벨을 50으로 하드코딩하지만, Pine Script는 `respectrsilevel` 파라미터(기본값 50)로 설정 가능하다

### Expected Behavior (Correct)

**🔴 CRITICAL**

2.1 WHEN RSILimitConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 SHALL `upper=40, lower=60`을 기본값으로 사용하고 `rsi >= upper` → Long, `rsi <= lower` → Short 로직을 적용하여 Pine Script와 동일한 시그널을 생성한다

2.2 WHEN RSIMALimitConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 SHALL `upper=40, lower=60`을 기본값으로 사용하고 `rsi_ma >= upper` → Long, `rsi_ma <= lower` → Short 로직을 적용하여 Pine Script와 동일한 시그널을 생성한다

**🟠 MAJOR**

2.3 WHEN BullBearPowerTrendConfirmation이 실행될 때 THEN 시스템은 SHALL Pine Script와 동일하게 `BullTrend = (close - lowest(low, 50)) / ATR(5)`, `BearTrend = (highest(high, 50) - close) / ATR(5)`, `BearTrend2 = -1 * BearTrend`, `Trend = BullTrend - BearTrend` 공식을 사용하고, Follow Trend 서브타입에서 `BearTrend_hist > 0 AND Trend >= 2` → Long, Without Trend 서브타입에서 `BearTrend_hist > 0` → Long 로직을 적용한다

2.4 WHEN CCIConfirmation이 실행될 때 THEN 시스템은 SHALL `upper_band=100, lower_band=-100` 파라미터를 추가하고 `cci > upper_band` → Long, `cci < lower_band` → Short 로직을 적용한다

2.5 WHEN IchimokuCloudConfirmation이 실행될 때 THEN 시스템은 SHALL Pine Script와 동일하게 5개 조건을 동시에 확인한다: `conversionLine > baseLine AND leadLine1 > leadLine2 AND close > leadLine1[displacement-1] AND close > leadLine2[displacement-1] AND ChikouSpan > leadLine1[50] AND ChikouSpan > leadLine2[50]`

2.6 WHEN SuperIchiConfirmation이 실행될 때 THEN 시스템은 SHALL Pine Script와 동일하게 ATR 기반 trailing stop `avg()` 함수를 구현하고 `tenkan_mult=2, kijun_mult=4, spanB_mult=6` 파라미터를 사용하며, `tenkan > kijun AND senkouA > senkouB AND close > senkouA[displacement-1] AND close > senkouB[displacement-1] AND ChikouSpan > senkouA[50] AND ChikouSpan > senkouB[50]` 조건을 적용한다

2.7 WHEN TSIConfirmation이 "Zero line cross" 서브타입으로 실행될 때 THEN 시스템은 SHALL `tsi > signal AND tsi > 0` 두 조건을 동시에 확인하여 Long 시그널을 생성한다 (Short: `tsi < signal AND tsi < 0`)

2.8 WHEN BXtrenderConfirmation이 실행될 때 THEN 시스템은 SHALL Pine Script와 동일하게 `ta.rsi(ema_diff, short_l3)` + T3 이동평균 기반 계산을 사용하고, `long_l1=5, long_l2=10` 파라미터를 추가하며, Short and Long term 서브타입에서 `maShortTermXtrender > maShortTermXtrender[1] AND longTermXtrender > 0 AND longTermXtrender > longTermXtrender[1] AND shortTermXtrender > shortTermXtrender[1] AND shortTermXtrender > 0` 조건을 적용한다

2.9 WHEN WaddahAttarExplosionConfirmation이 실행될 때 THEN 시스템은 SHALL deadzone 계산(`RMA(TR, 100) * 3.7`)을 추가하고 `trendUp > e1 AND e1 > deadzone AND trendUp > deadzone` → Long, `trendDown > e1 AND e1 > deadzone AND trendDown > deadzone` → Short 조건을 적용한다

2.10 WHEN DonchianTrendRibbonConfirmation이 실행될 때 THEN 시스템은 SHALL Pine Script와 동일하게 단일 `dchannel(dlen)` 함수를 구현하여 `close > highest[1]` → trend=1, `close < lowest[1]` → trend=-1 브레이크아웃 기반 로직을 적용한다

**🟡 MODERATE**

2.11 WHEN DMIADXConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 SHALL `length=10, adx_smoothing=5`를 기본값으로 사용하고, Advance 서브타입에서 adxcycle 및 DI 차이 > 1 조건을 포함한다

2.12 WHEN ChoppinessIndexConfirmation이 기본 파라미터로 실행될 때 THEN 시스템은 SHALL 단일 `ci_limit=61.8` 파라미터를 사용하고 `ci < ci_limit` 조건으로 추세 필터를 적용한다

2.13 WHEN RSIMADirectionConfirmation이 실행될 때 THEN 시스템은 SHALL `rsi_ma >= rsi_ma.shift(1)` (`>=`) → Long, `rsi_ma <= rsi_ma.shift(1)` (`<=`) → Short 비교를 사용한다

**🔵 MINOR**

2.14 WHEN RSIConfirmation이 "RSI Level" 서브타입으로 실행될 때 THEN 시스템은 SHALL `level` 파라미터(기본값 50)를 `default_params`에 추가하고 `rsi > level` → Long, `rsi < level` → Short 로직을 적용한다

### Unchanged Behavior (Regression Prevention)

3.1 WHEN 수정 대상이 아닌 confirmation indicator(EMAFilter, TwoEMACross, ThreeEMACross, RangeFilter, RQK, Supertrend, HalfTrend, ROC, McGinleyDynamic, DPO, BBOscillator, Stochastic, MACD, TrendlineBreakout, RangeDetector, HACOLT, ChandelierExit, ParabolicSAR, SSLChannel, HullSuite, AwesomeOscillator, VolatilityOscillator, DamianiVolatility, Volume, WolfpackId, QQEMod, ChaikinMoneyFlow, VortexIndicator, STC, VWAP)가 실행될 때 THEN 시스템은 SHALL CONTINUE TO 기존과 동일한 시그널을 생성한다

3.2 WHEN RSIConfirmation이 "RSI MA Cross" 또는 "RSI Exits OB-OS" 서브타입으로 실행될 때 THEN 시스템은 SHALL CONTINUE TO 기존과 동일한 시그널을 생성한다

3.3 WHEN TSIConfirmation이 "Signal Cross" 서브타입으로 실행될 때 THEN 시스템은 SHALL CONTINUE TO 기존과 동일한 시그널을 생성한다

3.4 WHEN DMIADXConfirmation이 "Adx Only" 또는 "Adx & +Di -Di" 서브타입으로 실행될 때 THEN 시스템은 SHALL CONTINUE TO 기본값 변경을 제외하고 동일한 로직으로 시그널을 생성한다

3.5 WHEN 수정된 indicator에 사용자 지정 파라미터가 전달될 때 THEN 시스템은 SHALL CONTINUE TO `_resolve_params` 메서드를 통해 사용자 파라미터를 기본값에 병합하여 적용한다

3.6 WHEN 수정된 indicator에 유효하지 않은 서브타입이 전달될 때 THEN 시스템은 SHALL CONTINUE TO `ValueError`를 발생시킨다

3.7 WHEN 수정된 indicator의 결과가 반환될 때 THEN 시스템은 SHALL CONTINUE TO `IndicatorResult(long_signal, short_signal)` 형식으로 NaN을 False로 채운 boolean Series를 반환한다
