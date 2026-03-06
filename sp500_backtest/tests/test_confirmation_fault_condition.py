"""14개 Confirmation Indicator Pine Script 불일치 버그 조건 탐색 테스트.

수정 전 코드에서 실행하여 버그 존재를 확인한다.
각 테스트는 Pine Script의 기대 동작을 인코딩하며,
수정 전 코드에서는 FAIL, 수정 후 코드에서는 PASS해야 한다.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14**
"""

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.indicators.confirmation import (
    BullBearPowerTrendConfirmation,
    BXtrenderConfirmation,
    CCIConfirmation,
    ChoppinessIndexConfirmation,
    DMIADXConfirmation,
    DonchianTrendRibbonConfirmation,
    IchimokuCloudConfirmation,
    RSIConfirmation,
    RSILimitConfirmation,
    RSIMADirectionConfirmation,
    RSIMALimitConfirmation,
    SuperIchiConfirmation,
    TSIConfirmation,
    WaddahAttarExplosionConfirmation,
)


def _make_ohlcv(
    n: int = 200,
    base_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """합성 OHLCV DataFrame 생성.

    Args:
        n: 데이터 포인트 수.
        base_price: 시작 가격.
        trend: 일일 추세 (양수=상승, 음수=하락).
        volatility: 가격 변동성 스케일.
        seed: 랜덤 시드 (재현성).

    Returns:
        OHLCV DataFrame (Open, High, Low, Close, Volume 컬럼).
    """
    rng = np.random.RandomState(seed)  # 재현 가능한 랜덤 생성기
    closes = [base_price]
    for _ in range(1, n):
        change = trend + volatility * rng.randn()  # 일일 변화량
        closes.append(max(closes[-1] + change, 1.0))  # 최소 가격 1.0 보장

    close = np.array(closes)
    # High/Low는 Close 기준 ±변동성으로 생성
    high = close + np.abs(rng.randn(n)) * volatility * 0.5
    low = close - np.abs(rng.randn(n)) * volatility * 0.5
    low = np.maximum(low, 0.5)  # 최소 가격 보장
    # Open은 이전 Close 근처
    open_ = np.roll(close, 1) + rng.randn(n) * volatility * 0.2
    open_[0] = base_price
    open_ = np.maximum(open_, 0.5)

    # OHLC 관계 보장: Low <= Open,Close <= High
    high = np.maximum(high, np.maximum(close, open_))
    low = np.minimum(low, np.minimum(close, open_))

    volume = np.abs(rng.randn(n)) * 1_000_000 + 500_000  # 거래량

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def _make_flat_ohlcv(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    """일정한 가격의 OHLCV DataFrame 생성 (RSI MA Direction 테스트용).

    Args:
        n: 데이터 포인트 수.
        price: 고정 가격.

    Returns:
        모든 OHLCV 값이 동일한 DataFrame.
    """
    return pd.DataFrame({
        "Open": np.full(n, price),
        "High": np.full(n, price + 0.01),
        "Low": np.full(n, price - 0.01),
        "Close": np.full(n, price),
        "Volume": np.full(n, 1_000_000.0),
    })


# ---------------------------------------------------------------------------
# 1. RSILimitConfirmation — 기본값 및 비교 연산자 버그
# **Validates: Requirements 1.1**
# ---------------------------------------------------------------------------


class TestRSILimitConfirmationFault:
    """RSILimitConfirmation 기본값 및 로직 버그 확인.

    Pine Script 기대값: upper=40, lower=60, rsi >= upper → Long, rsi <= lower → Short.
    현재 버그: upper=70, lower=30, rsi < upper → Long, rsi > lower → Short.
    """

    def test_default_params_upper_should_be_40(self):
        """기본 upper 파라미터가 40이어야 한다 (현재 70으로 설정됨)."""
        indicator = RSILimitConfirmation()
        assert indicator.default_params["upper"] == 40, (
            f"Pine Script 기본값은 rsilimitup=40이지만, "
            f"현재 upper={indicator.default_params['upper']}"
        )

    def test_default_params_lower_should_be_60(self):
        """기본 lower 파라미터가 60이어야 한다 (현재 30으로 설정됨)."""
        indicator = RSILimitConfirmation()
        assert indicator.default_params["lower"] == 60, (
            f"Pine Script 기본값은 rsilimitdown=60이지만, "
            f"현재 lower={indicator.default_params['lower']}"
        )

    def test_long_uses_gte_comparison(self):
        """rsi >= upper → Long 로직 확인 (현재 rsi < upper 사용).

        RSI가 정확히 upper(40)와 같을 때 Long=True여야 한다.
        """
        indicator = RSILimitConfirmation()
        # 하락 추세 데이터로 RSI를 40 근처로 만듦
        df = _make_ohlcv(n=200, trend=-0.3, volatility=2.0, seed=100)
        result = indicator.calculate(df, params={"upper": 40, "lower": 60})

        # RSI >= 40인 지점에서 Long=True여야 함
        from sp500_backtest.indicators.utils import rsi as rsi_func

        rsi_val = rsi_func(df["Close"], 14)
        # RSI가 40 이상인 지점 찾기
        mask_gte_40 = rsi_val >= 40
        valid_mask = mask_gte_40 & ~rsi_val.isna()

        if valid_mask.any():
            # rsi >= 40인 모든 지점에서 Long=True여야 함
            assert result.long_signal[valid_mask].all(), (
                "rsi >= upper(40)인 지점에서 Long=True여야 하지만 False인 지점이 있음. "
                "현재 코드는 rsi < upper 비교를 사용하고 있음."
            )


# ---------------------------------------------------------------------------
# 2. RSIMALimitConfirmation — 기본값 및 비교 연산자 버그
# **Validates: Requirements 1.2**
# ---------------------------------------------------------------------------


class TestRSIMALimitConfirmationFault:
    """RSIMALimitConfirmation 기본값 및 로직 버그 확인.

    Pine Script 기대값: upper=40, lower=60, rsi_ma >= upper → Long.
    현재 버그: upper=70, lower=30, rsi_ma < upper → Long.
    """

    def test_default_params_upper_should_be_40(self):
        """기본 upper 파라미터가 40이어야 한다 (현재 70으로 설정됨)."""
        indicator = RSIMALimitConfirmation()
        assert indicator.default_params["upper"] == 40, (
            f"Pine Script 기본값은 rsimalimitup=40이지만, "
            f"현재 upper={indicator.default_params['upper']}"
        )

    def test_default_params_lower_should_be_60(self):
        """기본 lower 파라미터가 60이어야 한다 (현재 30으로 설정됨)."""
        indicator = RSIMALimitConfirmation()
        assert indicator.default_params["lower"] == 60, (
            f"Pine Script 기본값은 rsimalimitdown=60이지만, "
            f"현재 lower={indicator.default_params['lower']}"
        )

    def test_long_uses_gte_comparison(self):
        """rsi_ma >= upper → Long 로직 확인 (현재 rsi_ma < upper 사용).

        RSI MA가 upper(40) 이상일 때 Long=True여야 한다.
        """
        indicator = RSIMALimitConfirmation()
        df = _make_ohlcv(n=200, trend=-0.3, volatility=2.0, seed=200)
        result = indicator.calculate(df, params={"upper": 40, "lower": 60})

        from sp500_backtest.indicators.utils import ma, rsi as rsi_func

        rsi_val = rsi_func(df["Close"], 14)
        rsi_ma = ma(rsi_val, 14, "SMA")

        # rsi_ma >= 40인 지점에서 Long=True여야 함
        mask_gte_40 = rsi_ma >= 40
        valid_mask = mask_gte_40 & ~rsi_ma.isna()

        if valid_mask.any():
            assert result.long_signal[valid_mask].all(), (
                "rsi_ma >= upper(40)인 지점에서 Long=True여야 하지만 False인 지점이 있음. "
                "현재 코드는 rsi_ma < upper 비교를 사용하고 있음."
            )


# ---------------------------------------------------------------------------
# 3. BullBearPowerTrendConfirmation — 공식 완전 불일치
# **Validates: Requirements 1.3**
# ---------------------------------------------------------------------------


class TestBullBearPowerTrendConfirmationFault:
    """BullBearPowerTrendConfirmation 공식 버그 확인.

    Pine Script 기대값: BullTrend = (close - lowest(low, 50)) / ATR(5).
    현재 버그: bull = high - EMA(close, 50) (Elder 공식).
    """

    def test_formula_uses_normalized_bulltrend(self):
        """Pine Script 공식: (close - lowest(low, 50)) / ATR(5) 사용 확인.

        Follow Trend 서브타입에서 BearTrend_hist > 0 AND Trend >= 2 → Long.
        현재 코드는 Elder 공식(high - EMA)을 사용하므로 다른 결과를 생성한다.
        """
        indicator = BullBearPowerTrendConfirmation()
        # 강한 상승 추세 데이터
        df = _make_ohlcv(n=200, trend=0.5, volatility=1.5, seed=300)

        from sp500_backtest.indicators.utils import atr as atr_func

        close = df["Close"]
        low = df["Low"]
        high = df["High"]

        # Pine Script 기대 공식
        lowest_low = low.rolling(window=50, min_periods=50).min()
        highest_high = high.rolling(window=50, min_periods=50).max()
        atr_val = atr_func(df, 5)

        bull_trend = (close - lowest_low) / atr_val.replace(0, np.nan)
        bear_trend = (highest_high - close) / atr_val.replace(0, np.nan)
        bear_trend2 = -1 * bear_trend

        # 히스토그램 계산
        bear_trend_hist = pd.Series(0.0, index=df.index)
        mask_bear = bear_trend2 > -2
        bear_trend_hist[mask_bear] = bear_trend2[mask_bear] + 2

        trend = bull_trend - bear_trend

        # Follow Trend: BearTrend_hist > 0 AND Trend >= 2 → Long
        expected_long = (bear_trend_hist > 0) & (trend >= 2)
        expected_long = expected_long.fillna(False)

        result = indicator.calculate(df, subtype="Follow Trend")

        # 유효한 데이터 범위 (워밍업 이후)
        valid_idx = expected_long.index[60:]
        if len(valid_idx) > 0:
            # 현재 코드는 Elder 공식을 사용하므로 Pine Script 기대값과 다를 것
            matches = (result.long_signal[valid_idx] == expected_long[valid_idx]).all()
            assert matches, (
                "BullBearPowerTrend Follow Trend 시그널이 Pine Script 기대값과 불일치. "
                "현재 코드는 Elder 공식(high - EMA)을 사용하지만, "
                "Pine Script는 (close - lowest(low,50))/ATR(5) 공식을 사용."
            )


# ---------------------------------------------------------------------------
# 4. CCIConfirmation — 밴드 기반 비교 버그
# **Validates: Requirements 1.4**
# ---------------------------------------------------------------------------


class TestCCIConfirmationFault:
    """CCIConfirmation 밴드 기반 비교 버그 확인.

    Pine Script 기대값: cci > 100 → Long, cci < -100 → Short.
    현재 버그: cci > 0 → Long, cci < 0 → Short.
    """

    def test_long_requires_cci_above_100(self):
        """CCI가 0~100 사이일 때 Long=False여야 한다.

        현재 코드는 cci > 0이면 Long=True를 반환하지만,
        Pine Script는 cci > 100이어야 Long=True.
        """
        indicator = CCIConfirmation()
        df = _make_ohlcv(n=200, trend=0.1, volatility=1.0, seed=400)
        result = indicator.calculate(df)

        # CCI 직접 계산
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tp = (high + low + close) / 3.0
        from sp500_backtest.indicators.utils import ma

        tp_sma = ma(tp, 20, "SMA")
        mad = tp.rolling(window=20, min_periods=20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        cci = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))

        # CCI가 0~100 사이인 지점 찾기 (Long=False여야 함)
        between_0_100 = (cci > 0) & (cci <= 100) & ~cci.isna()
        if between_0_100.any():
            # Pine Script에서는 이 범위에서 Long=False
            assert not result.long_signal[between_0_100].any(), (
                "CCI가 0~100 사이일 때 Long=False여야 하지만 True인 지점이 있음. "
                "현재 코드는 cci > 0 비교를 사용하지만, "
                "Pine Script는 cci > 100(upper_band) 비교를 사용."
            )

    def test_short_requires_cci_below_minus_100(self):
        """CCI가 -100~0 사이일 때 Short=False여야 한다."""
        indicator = CCIConfirmation()
        df = _make_ohlcv(n=200, trend=-0.1, volatility=1.0, seed=401)
        result = indicator.calculate(df)

        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tp = (high + low + close) / 3.0
        from sp500_backtest.indicators.utils import ma

        tp_sma = ma(tp, 20, "SMA")
        mad = tp.rolling(window=20, min_periods=20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        cci = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))

        # CCI가 -100~0 사이인 지점 찾기 (Short=False여야 함)
        between_neg100_0 = (cci < 0) & (cci >= -100) & ~cci.isna()
        if between_neg100_0.any():
            assert not result.short_signal[between_neg100_0].any(), (
                "CCI가 -100~0 사이일 때 Short=False여야 하지만 True인 지점이 있음. "
                "현재 코드는 cci < 0 비교를 사용하지만, "
                "Pine Script는 cci < -100(lower_band) 비교를 사용."
            )


# ---------------------------------------------------------------------------
# 5. IchimokuCloudConfirmation — 5개 조건 동시 충족 버그
# **Validates: Requirements 1.5**
# ---------------------------------------------------------------------------


class TestIchimokuCloudConfirmationFault:
    """IchimokuCloudConfirmation 조건 수 불일치 버그 확인.

    Pine Script 기대값: 5개 조건 동시 충족 (전환선>기준선, 선행스팬A>B 등).
    현재 버그: close > cloud_top 단일 조건만 확인.
    """

    def test_requires_five_conditions_simultaneously(self):
        """close > cloud_top이지만 전환선 < 기준선일 때 Long=False여야 한다.

        현재 코드는 close > cloud_top만 확인하므로 Long=True를 반환하지만,
        Pine Script는 전환선 > 기준선 조건도 필요.
        """
        indicator = IchimokuCloudConfirmation()
        df = _make_ohlcv(n=200, trend=0.2, volatility=2.0, seed=500)

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # 이치모쿠 계산
        tenkan_h = high.rolling(9, min_periods=9).max()
        tenkan_l = low.rolling(9, min_periods=9).min()
        tenkan = (tenkan_h + tenkan_l) / 2

        kijun_h = high.rolling(26, min_periods=26).max()
        kijun_l = low.rolling(26, min_periods=26).min()
        kijun = (kijun_h + kijun_l) / 2

        senkou_a = (tenkan + kijun) / 2
        senkou_b_h = high.rolling(52, min_periods=52).max()
        senkou_b_l = low.rolling(52, min_periods=52).min()
        senkou_b = (senkou_b_h + senkou_b_l) / 2

        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)

        # close > cloud_top이지만 tenkan < kijun인 지점 찾기
        above_cloud = close > cloud_top
        tenkan_below_kijun = tenkan < kijun
        bug_condition = above_cloud & tenkan_below_kijun & ~cloud_top.isna()

        result = indicator.calculate(df)

        if bug_condition.any():
            # Pine Script에서는 이 지점에서 Long=False (전환선 < 기준선이므로)
            assert not result.long_signal[bug_condition].any(), (
                "close > cloud_top이지만 전환선 < 기준선일 때 Long=False여야 함. "
                "현재 코드는 close > cloud_top 단일 조건만 확인하지만, "
                "Pine Script는 5개 조건을 동시에 확인."
            )


# ---------------------------------------------------------------------------
# 6. SuperIchiConfirmation — ATR 기반 계산 불일치
# **Validates: Requirements 1.6**
# ---------------------------------------------------------------------------


class TestSuperIchiConfirmationFault:
    """SuperIchiConfirmation ATR 기반 계산 불일치 버그 확인.

    Pine Script 기대값: ATR trailing stop avg() + tenkan_mult=2, kijun_mult=4, spanB_mult=6.
    현재 버그: 표준 donchian 계산 사용, multiplier 파라미터 없음.
    """

    def test_default_params_should_have_multipliers(self):
        """기본 파라미터에 tenkan_mult, kijun_mult, spanB_mult가 있어야 한다."""
        indicator = SuperIchiConfirmation()
        params = indicator.default_params

        assert "tenkan_mult" in params or "tenkan" not in params, (
            "SuperIchi는 ATR 기반 계산을 사용하므로 tenkan_mult 파라미터가 필요. "
            f"현재 파라미터: {list(params.keys())}"
        )

        # 더 구체적으로: tenkan_mult=2.0이 있어야 함
        if "tenkan_mult" in params:
            assert params["tenkan_mult"] == 2.0
        else:
            # tenkan_mult가 없으면 버그
            pytest.fail(
                "SuperIchi default_params에 tenkan_mult 파라미터가 없음. "
                "Pine Script는 tenkan_mult=2, kijun_mult=4, spanB_mult=6을 사용."
            )

    def test_default_params_should_have_displacement(self):
        """기본 파라미터에 displacement=26이 있어야 한다."""
        indicator = SuperIchiConfirmation()
        params = indicator.default_params

        assert "displacement" in params, (
            "SuperIchi default_params에 displacement 파라미터가 없음. "
            "Pine Script는 displacement=26을 사용."
        )


# ---------------------------------------------------------------------------
# 7. TSIConfirmation (Zero line cross) — 두 조건 동시 확인 버그
# **Validates: Requirements 1.7**
# ---------------------------------------------------------------------------


class TestTSIConfirmationFault:
    """TSIConfirmation Zero line cross 서브타입 버그 확인.

    Pine Script 기대값: tsi > signal AND tsi > 0 → Long.
    현재 버그: tsi > 0 단일 조건만 확인.
    """

    def test_zero_line_cross_requires_tsi_above_signal(self):
        """tsi > 0이지만 tsi < signal일 때 Long=False여야 한다.

        현재 코드는 tsi > 0만 확인하므로 Long=True를 반환하지만,
        Pine Script는 tsi > signal도 동시에 요구.
        """
        indicator = TSIConfirmation()
        df = _make_ohlcv(n=200, trend=0.05, volatility=1.5, seed=700)

        from sp500_backtest.indicators.utils import ma

        close = df["Close"]
        momentum = close.diff()
        double_smooth_mom = ma(ma(momentum, 25, "EMA"), 13, "EMA")
        double_smooth_abs = ma(ma(momentum.abs(), 25, "EMA"), 13, "EMA")
        tsi = 100.0 * double_smooth_mom / double_smooth_abs.replace(0, np.nan)
        signal_line = ma(tsi, 13, "EMA")

        # tsi > 0이지만 tsi < signal인 지점 찾기
        tsi_pos_below_signal = (tsi > 0) & (tsi < signal_line) & ~tsi.isna() & ~signal_line.isna()

        result = indicator.calculate(df, subtype="Zero line cross")

        if tsi_pos_below_signal.any():
            assert not result.long_signal[tsi_pos_below_signal].any(), (
                "tsi > 0이지만 tsi < signal일 때 Long=False여야 함. "
                "현재 코드는 tsi > 0 단일 조건만 확인하지만, "
                "Pine Script는 tsi > signal AND tsi > 0 두 조건을 동시에 요구."
            )


# ---------------------------------------------------------------------------
# 8. BXtrenderConfirmation — RSI+T3 기반 계산 및 파라미터 버그
# **Validates: Requirements 1.8**
# ---------------------------------------------------------------------------


class TestBXtrenderConfirmationFault:
    """BXtrenderConfirmation 계산 방식 및 파라미터 버그 확인.

    Pine Script 기대값: RSI + T3 이동평균, 방향 비교(> [1]), long_l1=5, long_l2=10.
    현재 버그: stoch + SMA, 레벨 비교(> 0), long_l1/long_l2 파라미터 누락.
    """

    def test_default_params_should_have_long_params(self):
        """기본 파라미터에 long_l1=5, long_l2=10이 있어야 한다."""
        indicator = BXtrenderConfirmation()
        params = indicator.default_params

        assert "long_l1" in params, (
            "BXtrender default_params에 long_l1 파라미터가 없음. "
            "Pine Script는 long_l1=5를 사용."
        )
        if "long_l1" in params:
            assert params["long_l1"] == 5

        assert "long_l2" in params, (
            "BXtrender default_params에 long_l2 파라미터가 없음. "
            "Pine Script는 long_l2=10을 사용."
        )
        if "long_l2" in params:
            assert params["long_l2"] == 10

    def test_subtypes_should_have_trend_suffix(self):
        """서브타입 이름에 'trend' 접미사가 있어야 한다.

        Pine Script: 'Short Term trend', 'Short and Long term trend'.
        현재: 'Short term', 'Short and Long term'.
        """
        indicator = BXtrenderConfirmation()
        subtypes = indicator.subtypes

        assert "Short Term trend" in subtypes, (
            f"서브타입에 'Short Term trend'가 없음. 현재 서브타입: {subtypes}. "
            "Pine Script는 'Short Term trend'를 사용."
        )
        assert "Short and Long term trend" in subtypes, (
            f"서브타입에 'Short and Long term trend'가 없음. 현재 서브타입: {subtypes}. "
            "Pine Script는 'Short and Long term trend'를 사용."
        )


# ---------------------------------------------------------------------------
# 9. WaddahAttarExplosionConfirmation — deadzone 필터 누락
# **Validates: Requirements 1.9**
# ---------------------------------------------------------------------------


class TestWaddahAttarExplosionConfirmationFault:
    """WaddahAttarExplosionConfirmation deadzone 필터 누락 버그 확인.

    Pine Script 기대값: trendUp > e1 AND e1 > deadzone AND trendUp > deadzone → Long.
    현재 버그: trend > 0 AND trend > explosion → Long (deadzone 필터 없음).
    """

    def test_long_requires_deadzone_filter(self):
        """현재 코드에 deadzone 필터 로직이 없음을 확인.

        Pine Script는 deadzone = RMA(TR, 100) * 3.7 필터를 필수로 적용하지만,
        현재 코드는 deadzone 계산 자체가 없다.
        현재 코드: trend > 0 AND trend > explosion → Long (deadzone 없음)
        Pine Script: trendUp > e1 AND e1 > deadzone AND trendUp > deadzone → Long
        """
        indicator = WaddahAttarExplosionConfirmation()
        # 낮은 변동성 + 약한 추세 데이터 (deadzone이 explosion보다 클 가능성 높음)
        df = _make_ohlcv(n=250, trend=0.05, volatility=0.3, seed=901)

        from sp500_backtest.indicators.utils import ma, true_range

        close = df["Close"]

        # MACD 기반 트렌드 (현재 코드와 동일)
        macd1 = ma(close, 20, "EMA") - ma(close, 40, "EMA")
        macd2 = ma(close.shift(1), 20, "EMA") - ma(close.shift(1), 40, "EMA")
        trend = (macd1 - macd2) * 150
        trend_up = trend.clip(lower=0)

        # 볼린저 밴드 기반 폭발선
        bb_sma = ma(close, 20, "SMA")
        bb_std = close.rolling(window=20, min_periods=20).std()
        e1 = (bb_sma + 2.0 * bb_std) - (bb_sma - 2.0 * bb_std)

        # Pine Script deadzone 계산
        tr = true_range(df)
        deadzone = ma(tr, 100, "RMA") * 3.7

        # 현재 코드가 Long=True를 반환하지만 Pine Script에서는 Long=False인 지점 찾기
        # 현재 코드: trend > 0 AND trend > explosion
        current_long = (trend > 0) & (trend > e1)
        # Pine Script: trendUp > e1 AND e1 > deadzone AND trendUp > deadzone
        pine_long = (trend_up > e1) & (e1 > deadzone) & (trend_up > deadzone)

        valid = ~trend.isna() & ~e1.isna() & ~deadzone.isna()
        # 현재 코드는 Long이지만 Pine Script는 Long이 아닌 지점 (deadzone 필터 때문)
        false_positive = current_long & ~pine_long & valid

        result = indicator.calculate(df)

        if false_positive.any():
            # 현재 코드가 deadzone 없이 Long을 반환하는 지점에서 실제로 Long인지 확인
            assert not result.long_signal[false_positive].any(), (
                "현재 코드가 deadzone 필터 없이 Long=True를 반환하는 지점이 있음. "
                "Pine Script는 e1 > deadzone 조건을 필수로 적용."
            )
        else:
            # 데이터에서 false positive가 없으면 직접 deadzone 존재 여부 확인
            # 현재 코드의 소스에 deadzone 관련 로직이 없음을 간접 확인
            import inspect
            source = inspect.getsource(indicator._calculate_impl)
            assert "deadzone" in source, (
                "WaddahAttar _calculate_impl에 deadzone 로직이 없음. "
                "Pine Script는 RMA(TR, 100) * 3.7 기반 deadzone 필터를 필수로 적용."
            )


# ---------------------------------------------------------------------------
# 10. DonchianTrendRibbonConfirmation — 브레이크아웃 로직 불일치
# **Validates: Requirements 1.10**
# ---------------------------------------------------------------------------


class TestDonchianTrendRibbonConfirmationFault:
    """DonchianTrendRibbonConfirmation 로직 불일치 버그 확인.

    Pine Script 기대값: close > highest[1] → trend=1 (단일 브레이크아웃).
    현재 버그: 5개 기간 Donchian 중간값 합산 방식.
    """

    def test_breakout_logic_not_midpoint_sum(self):
        """Pine Script 브레이크아웃 로직과 현재 중간값 합산 로직이 다른 결과를 생성.

        Pine Script: close > highest(high, period)[1] → trend=1 → Long.
        현재: 5개 기간 중간값 합산 > 0 → Long.
        """
        indicator = DonchianTrendRibbonConfirmation()
        df = _make_ohlcv(n=200, trend=0.3, volatility=2.0, seed=1000)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        period = 15

        # Pine Script 기대 로직: 단일 브레이크아웃
        hh = high.rolling(window=period, min_periods=period).max()
        ll = low.rolling(window=period, min_periods=period).min()

        # 상태 기반 trend 계산
        n = len(df)
        trend_arr = np.zeros(n)
        for i in range(1, n):
            if np.isnan(hh.iloc[i]) or np.isnan(ll.iloc[i]):
                trend_arr[i] = trend_arr[i - 1]
                continue
            prev_hh = hh.iloc[i - 1] if i > 0 and not np.isnan(hh.iloc[i - 1]) else np.nan
            prev_ll = ll.iloc[i - 1] if i > 0 and not np.isnan(ll.iloc[i - 1]) else np.nan

            if not np.isnan(prev_hh) and close.iloc[i] > prev_hh:
                trend_arr[i] = 1
            elif not np.isnan(prev_ll) and close.iloc[i] < prev_ll:
                trend_arr[i] = -1
            else:
                trend_arr[i] = trend_arr[i - 1]

        expected_long = pd.Series(trend_arr == 1, index=df.index)
        expected_short = pd.Series(trend_arr == -1, index=df.index)

        result = indicator.calculate(df)

        # 유효한 범위 (워밍업 이후)
        valid_idx = df.index[80:]
        if len(valid_idx) > 0:
            long_match = (result.long_signal[valid_idx] == expected_long[valid_idx]).all()
            assert long_match, (
                "DonchianTrendRibbon Long 시그널이 Pine Script 브레이크아웃 로직과 불일치. "
                "현재 코드는 5개 기간 중간값 합산 방식을 사용하지만, "
                "Pine Script는 close > highest[1] 브레이크아웃 로직을 사용."
            )


# ---------------------------------------------------------------------------
# 11. DMIADXConfirmation — 기본값 및 Advance 서브타입 버그
# **Validates: Requirements 1.11**
# ---------------------------------------------------------------------------


class TestDMIADXConfirmationFault:
    """DMIADXConfirmation 기본값 및 Advance 서브타입 버그 확인.

    Pine Script 기대값: length=10, adx_smoothing=5, Advance에 adxcycle + DI diff > 1.
    현재 버그: length=14, adx_smoothing=14, Advance에 adxcycle 없음.
    """

    def test_default_length_should_be_10(self):
        """기본 length 파라미터가 10이어야 한다 (현재 14)."""
        indicator = DMIADXConfirmation()
        assert indicator.default_params["length"] == 10, (
            f"Pine Script 기본값은 dilen=10이지만, "
            f"현재 length={indicator.default_params['length']}"
        )

    def test_default_adx_smoothing_should_be_5(self):
        """기본 adx_smoothing 파라미터가 5여야 한다 (현재 14)."""
        indicator = DMIADXConfirmation()
        assert indicator.default_params["adx_smoothing"] == 5, (
            f"Pine Script 기본값은 adxlen=5이지만, "
            f"현재 adx_smoothing={indicator.default_params['adx_smoothing']}"
        )

    def test_advance_requires_di_diff_gt_1(self):
        """Advance 서브타입에서 adxcycle 상태 머신이 필요.

        현재 코드는 +DI > -DI AND ADX 상승만 확인하지만,
        Pine Script는 adxcycle 상태 머신 + diplus - diminus > 1 조건을 사용.
        현재 코드의 소스에 adxcycle 로직이 없음을 확인.
        """
        indicator = DMIADXConfirmation()

        # 현재 코드의 Advance 로직에 adxcycle이 없음을 직접 확인
        import inspect
        source = inspect.getsource(indicator._calculate_impl)

        assert "adxcycle" in source, (
            "DMI ADX Advance 서브타입에 adxcycle 상태 머신이 없음. "
            "Pine Script는 adxcycle 상태 변수(adx crossover keyLevel → 1, "
            "adx crossunder keyLevel → -1)와 DI 차이 > 1 조건을 사용."
        )


# ---------------------------------------------------------------------------
# 12. ChoppinessIndexConfirmation — 단일 임계값 버그
# **Validates: Requirements 1.12**
# ---------------------------------------------------------------------------


class TestChoppinessIndexConfirmationFault:
    """ChoppinessIndexConfirmation 이중 임계값 버그 확인.

    Pine Script 기대값: 단일 ci_limit=61.8, ci < ci_limit → 추세.
    현재 버그: trending_threshold=38.2, choppy_threshold=61.8 이중 임계값.
    """

    def test_default_params_should_have_ci_limit(self):
        """기본 파라미터에 ci_limit=61.8이 있어야 한다 (trending_threshold 대신)."""
        indicator = ChoppinessIndexConfirmation()
        params = indicator.default_params

        assert "ci_limit" in params, (
            "ChoppinessIndex default_params에 ci_limit 파라미터가 없음. "
            f"현재 파라미터: {list(params.keys())}. "
            "Pine Script는 단일 ci_limit=61.8을 사용."
        )

    def test_trending_threshold_should_not_be_38_2(self):
        """trending_threshold=38.2가 아닌 ci_limit=61.8을 사용해야 한다.

        CI가 38.2~61.8 사이일 때 현재 코드는 Long=False이지만,
        Pine Script는 ci < 61.8이면 Long=True.
        """
        indicator = ChoppinessIndexConfirmation()
        df = _make_ohlcv(n=200, trend=0.2, volatility=1.5, seed=1200)

        from sp500_backtest.indicators.utils import atr as atr_func

        high = df["High"]
        low = df["Low"]
        length = 14

        atr_val = atr_func(df, 1)
        atr_sum = atr_val.rolling(window=length, min_periods=length).sum()
        hh = high.rolling(window=length, min_periods=length).max()
        ll = low.rolling(window=length, min_periods=length).min()
        hl_range = (hh - ll).replace(0, np.nan)
        ci = 100.0 * np.log10(atr_sum / hl_range) / np.log10(length)

        # CI가 38.2~61.8 사이인 지점 (Pine Script에서는 Long=True)
        between_thresholds = (ci > 38.2) & (ci < 61.8) & ~ci.isna()

        result = indicator.calculate(df)

        if between_thresholds.any():
            # Pine Script: ci < 61.8 → Long=True
            assert result.long_signal[between_thresholds].all(), (
                "CI가 38.2~61.8 사이일 때 Long=True여야 함 (ci < ci_limit=61.8). "
                "현재 코드는 ci < trending_threshold(38.2)를 사용하므로 False를 반환."
            )


# ---------------------------------------------------------------------------
# 13. RSIMADirectionConfirmation — 비교 연산자 버그
# **Validates: Requirements 1.13**
# ---------------------------------------------------------------------------


class TestRSIMADirectionConfirmationFault:
    """RSIMADirectionConfirmation 비교 연산자 버그 확인.

    Pine Script 기대값: rsi_ma >= rsi_ma[1] → Long (equal 포함).
    현재 버그: rsi_ma > rsi_ma[1] → Long (strict, equal 미포함).
    """

    def test_gte_comparison_includes_equal(self):
        """RSI MA가 이전 값과 동일할 때 Long=True여야 한다.

        현재 코드는 strict > 비교를 사용하므로 False를 반환하지만,
        Pine Script는 >= 비교를 사용하므로 True여야 한다.
        """
        # 일정한 가격 데이터로 RSI MA가 변하지 않는 구간 생성
        df = _make_flat_ohlcv(n=200, price=100.0)
        indicator = RSIMADirectionConfirmation()
        result = indicator.calculate(df)

        from sp500_backtest.indicators.utils import ma, rsi as rsi_func

        rsi_val = rsi_func(df["Close"], 14)
        rsi_ma = ma(rsi_val, 14, "SMA")

        # RSI MA가 이전 값과 동일한 지점 찾기
        equal_mask = (rsi_ma == rsi_ma.shift(1)) & ~rsi_ma.isna() & ~rsi_ma.shift(1).isna()

        if equal_mask.any():
            # Pine Script: rsi_ma >= rsi_ma[1] → Long=True
            assert result.long_signal[equal_mask].all(), (
                "RSI MA가 이전 값과 동일할 때 Long=True여야 함 (>= 비교). "
                "현재 코드는 strict > 비교를 사용하므로 False를 반환."
            )


# ---------------------------------------------------------------------------
# 14. RSIConfirmation (RSI Level) — level 파라미터 하드코딩 버그
# **Validates: Requirements 1.14**
# ---------------------------------------------------------------------------


class TestRSIConfirmationLevelFault:
    """RSIConfirmation RSI Level 서브타입 파라미터 하드코딩 버그 확인.

    Pine Script 기대값: level 파라미터(기본값 50)로 설정 가능.
    현재 버그: 레벨 50 하드코딩.
    """

    def test_default_params_should_have_level(self):
        """기본 파라미터에 level=50이 있어야 한다."""
        indicator = RSIConfirmation()
        params = indicator.default_params

        assert "level" in params, (
            "RSIConfirmation default_params에 level 파라미터가 없음. "
            f"현재 파라미터: {list(params.keys())}. "
            "Pine Script는 respectrsilevel 파라미터(기본값 50)를 사용."
        )

    def test_level_param_is_configurable(self):
        """level 파라미터를 변경하면 시그널이 달라져야 한다.

        level=60으로 설정하면 RSI > 60 → Long이어야 하지만,
        현재 코드는 50 하드코딩이므로 level 파라미터를 무시.
        """
        indicator = RSIConfirmation()
        df = _make_ohlcv(n=200, trend=0.1, volatility=1.5, seed=1400)

        # level=60으로 설정
        result_60 = indicator.calculate(df, params={"level": 60}, subtype="RSI Level")
        # level=50으로 설정 (기본값)
        result_50 = indicator.calculate(df, params={"level": 50}, subtype="RSI Level")

        from sp500_backtest.indicators.utils import rsi as rsi_func

        rsi_val = rsi_func(df["Close"], 14)

        # RSI가 50~60 사이인 지점 찾기
        between_50_60 = (rsi_val > 50) & (rsi_val <= 60) & ~rsi_val.isna()

        if between_50_60.any():
            # level=50: Long=True, level=60: Long=False
            assert result_50.long_signal[between_50_60].any(), (
                "level=50일 때 RSI > 50인 지점에서 Long=True여야 함."
            )
            assert not result_60.long_signal[between_50_60].any(), (
                "level=60일 때 RSI가 50~60 사이인 지점에서 Long=False여야 함. "
                "현재 코드는 level 파라미터를 무시하고 50을 하드코딩."
            )
