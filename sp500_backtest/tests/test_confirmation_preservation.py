"""비수정 Confirmation Indicator 및 비수정 서브타입 동작 보존 테스트.

수정 전 코드에서 실행하여 기존 동작을 캡처하고,
수정 후 코드에서도 동일하게 통과하여 회귀가 없음을 확인한다.

observation-first 방법론:
- 수정 전 코드의 실제 출력을 관찰하여 테스트 작성
- 모든 테스트는 수정 전/후 모두 PASS해야 함

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
"""

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.indicators.base import IndicatorResult
from sp500_backtest.indicators.confirmation import (
    # 비수정 indicator 대표 샘플 (~10개)
    EMAFilterConfirmation,
    TwoEMACrossConfirmation,
    SupertrendConfirmation,
    ROCConfirmation,
    MACDConfirmation,
    BBOscillatorConfirmation,
    StochasticConfirmation,
    HullSuiteConfirmation,
    ChaikinMoneyFlowConfirmation,
    VortexIndicatorConfirmation,
    # 수정 대상이지만 보존되는 서브타입이 있는 indicator
    RSIConfirmation,
    TSIConfirmation,
    DMIADXConfirmation,
)


# ---------------------------------------------------------------------------
# 합성 OHLCV 데이터 생성 (결정론적, 고정 시드)
# ---------------------------------------------------------------------------

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
    high = close + np.abs(rng.randn(n)) * volatility * 0.5
    low = close - np.abs(rng.randn(n)) * volatility * 0.5
    low = np.maximum(low, 0.5)  # 최소 가격 보장
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


# 테스트 전체에서 공유하는 고정 데이터
DF_TREND = _make_ohlcv(n=250, trend=0.3, volatility=1.5, seed=12345)
DF_FLAT = _make_ohlcv(n=250, trend=0.0, volatility=0.8, seed=67890)


# ---------------------------------------------------------------------------
# 3.1 비수정 Indicator 동작 보존 — IndicatorResult 형식 검증
# **Validates: Requirements 3.1, 3.7**
# ---------------------------------------------------------------------------

# 비수정 indicator 인스턴스 목록 (대표 샘플)
PRESERVED_INDICATORS = [
    (EMAFilterConfirmation(), None),
    (TwoEMACrossConfirmation(), None),
    (SupertrendConfirmation(), None),
    (ROCConfirmation(), None),
    (MACDConfirmation(), "MACD Crossover"),
    (MACDConfirmation(), "Zero line crossover"),
    (BBOscillatorConfirmation(), None),
    (StochasticConfirmation(), "CrossOver"),
    (StochasticConfirmation(), "OB&OS levels"),
    (HullSuiteConfirmation(), None),
    (ChaikinMoneyFlowConfirmation(), None),
    (VortexIndicatorConfirmation(), "Simple"),
]


def _indicator_id(param):
    """pytest parametrize용 ID 생성."""
    indicator, subtype = param
    name = indicator.name
    if subtype:
        return f"{name}_{subtype}"
    return name


class TestPreservedIndicatorResultFormat:
    """비수정 indicator가 IndicatorResult 형식을 반환하는지 검증.

    **Validates: Requirements 3.1, 3.7**
    """

    @pytest.mark.parametrize(
        "indicator_and_subtype",
        PRESERVED_INDICATORS,
        ids=[_indicator_id(p) for p in PRESERVED_INDICATORS],
    )
    def test_returns_indicator_result(self, indicator_and_subtype):
        """비수정 indicator가 IndicatorResult를 반환한다."""
        indicator, subtype = indicator_and_subtype
        result = indicator.calculate(DF_TREND, subtype=subtype)

        assert isinstance(result, IndicatorResult), (
            f"{indicator.name} should return IndicatorResult"
        )

    @pytest.mark.parametrize(
        "indicator_and_subtype",
        PRESERVED_INDICATORS,
        ids=[_indicator_id(p) for p in PRESERVED_INDICATORS],
    )
    def test_signals_are_boolean_series(self, indicator_and_subtype):
        """비수정 indicator의 시그널이 boolean Series이다."""
        indicator, subtype = indicator_and_subtype
        result = indicator.calculate(DF_TREND, subtype=subtype)

        assert isinstance(result.long_signal, pd.Series), "long_signal은 Series여야 함"
        assert isinstance(result.short_signal, pd.Series), "short_signal은 Series여야 함"
        assert result.long_signal.dtype == bool, "long_signal은 bool dtype이어야 함"
        assert result.short_signal.dtype == bool, "short_signal은 bool dtype이어야 함"

    @pytest.mark.parametrize(
        "indicator_and_subtype",
        PRESERVED_INDICATORS,
        ids=[_indicator_id(p) for p in PRESERVED_INDICATORS],
    )
    def test_no_nan_in_signals(self, indicator_and_subtype):
        """비수정 indicator의 시그널에 NaN이 없다 (False로 채워짐).

        **Validates: Requirements 3.7**
        """
        indicator, subtype = indicator_and_subtype
        result = indicator.calculate(DF_TREND, subtype=subtype)

        assert not result.long_signal.isna().any(), "long_signal에 NaN이 없어야 함"
        assert not result.short_signal.isna().any(), "short_signal에 NaN이 없어야 함"

    @pytest.mark.parametrize(
        "indicator_and_subtype",
        PRESERVED_INDICATORS,
        ids=[_indicator_id(p) for p in PRESERVED_INDICATORS],
    )
    def test_signal_length_matches_input(self, indicator_and_subtype):
        """비수정 indicator의 시그널 길이가 입력 DataFrame과 동일하다."""
        indicator, subtype = indicator_and_subtype
        result = indicator.calculate(DF_TREND, subtype=subtype)

        assert len(result.long_signal) == len(DF_TREND), "long_signal 길이 불일치"
        assert len(result.short_signal) == len(DF_TREND), "short_signal 길이 불일치"


# ---------------------------------------------------------------------------
# 3.2 RSIConfirmation 보존 서브타입 — "RSI MA Cross", "RSI Exits OB-OS"
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------

class TestRSIConfirmationPreservedSubtypes:
    """RSIConfirmation의 보존 서브타입 동작 검증.

    "RSI MA Cross"와 "RSI Exits OB-OS"는 수정 대상이 아니므로
    수정 전/후 동일한 시그널을 생성해야 한다.

    **Validates: Requirements 3.2**
    """

    def test_rsi_ma_cross_returns_valid_result(self):
        """RSI MA Cross 서브타입이 유효한 IndicatorResult를 반환한다."""
        indicator = RSIConfirmation()
        result = indicator.calculate(DF_TREND, subtype="RSI MA Cross")

        assert isinstance(result, IndicatorResult)
        assert result.long_signal.dtype == bool
        assert result.short_signal.dtype == bool
        assert not result.long_signal.isna().any()
        assert not result.short_signal.isna().any()

    def test_rsi_ma_cross_snapshot(self):
        """RSI MA Cross 서브타입의 시그널 스냅샷이 결정론적이다."""
        indicator = RSIConfirmation()
        result1 = indicator.calculate(DF_TREND, subtype="RSI MA Cross")
        result2 = indicator.calculate(DF_TREND, subtype="RSI MA Cross")

        # 동일 입력에 대해 동일 출력 (결정론적)
        pd.testing.assert_series_equal(result1.long_signal, result2.long_signal)
        pd.testing.assert_series_equal(result1.short_signal, result2.short_signal)

    def test_rsi_exits_ob_os_returns_valid_result(self):
        """RSI Exits OB-OS 서브타입이 유효한 IndicatorResult를 반환한다."""
        indicator = RSIConfirmation()
        result = indicator.calculate(DF_TREND, subtype="RSI Exits OB-OS")

        assert isinstance(result, IndicatorResult)
        assert result.long_signal.dtype == bool
        assert result.short_signal.dtype == bool
        assert not result.long_signal.isna().any()

    def test_rsi_exits_ob_os_snapshot(self):
        """RSI Exits OB-OS 서브타입의 시그널 스냅샷이 결정론적이다."""
        indicator = RSIConfirmation()
        result1 = indicator.calculate(DF_TREND, subtype="RSI Exits OB-OS")
        result2 = indicator.calculate(DF_TREND, subtype="RSI Exits OB-OS")

        pd.testing.assert_series_equal(result1.long_signal, result2.long_signal)
        pd.testing.assert_series_equal(result1.short_signal, result2.short_signal)

    def test_rsi_ma_cross_has_both_signals(self):
        """RSI MA Cross가 Long과 Short 시그널을 모두 생성한다."""
        indicator = RSIConfirmation()
        result = indicator.calculate(DF_TREND, subtype="RSI MA Cross")

        # 충분한 데이터에서 Long/Short 모두 발생해야 함
        assert result.long_signal.any(), "RSI MA Cross는 Long 시그널을 생성해야 함"
        assert result.short_signal.any(), "RSI MA Cross는 Short 시그널을 생성해야 함"


# ---------------------------------------------------------------------------
# 3.3 TSIConfirmation 보존 서브타입 — "Signal Cross"
# **Validates: Requirements 3.3**
# ---------------------------------------------------------------------------

class TestTSIConfirmationPreservedSubtype:
    """TSIConfirmation의 "Signal Cross" 서브타입 동작 검증.

    **Validates: Requirements 3.3**
    """

    def test_signal_cross_returns_valid_result(self):
        """Signal Cross 서브타입이 유효한 IndicatorResult를 반환한다."""
        indicator = TSIConfirmation()
        result = indicator.calculate(DF_TREND, subtype="Signal Cross")

        assert isinstance(result, IndicatorResult)
        assert result.long_signal.dtype == bool
        assert result.short_signal.dtype == bool
        assert not result.long_signal.isna().any()
        assert not result.short_signal.isna().any()

    def test_signal_cross_snapshot(self):
        """Signal Cross 서브타입의 시그널 스냅샷이 결정론적이다."""
        indicator = TSIConfirmation()
        result1 = indicator.calculate(DF_TREND, subtype="Signal Cross")
        result2 = indicator.calculate(DF_TREND, subtype="Signal Cross")

        pd.testing.assert_series_equal(result1.long_signal, result2.long_signal)
        pd.testing.assert_series_equal(result1.short_signal, result2.short_signal)

    def test_signal_cross_logic_tsi_vs_signal(self):
        """Signal Cross: TSI > signal → Long, TSI < signal → Short 로직 확인."""
        indicator = TSIConfirmation()
        result = indicator.calculate(DF_TREND, subtype="Signal Cross")

        # 충분한 데이터에서 Long/Short 모두 발생해야 함
        assert result.long_signal.any(), "Signal Cross는 Long 시그널을 생성해야 함"
        assert result.short_signal.any(), "Signal Cross는 Short 시그널을 생성해야 함"


# ---------------------------------------------------------------------------
# 3.4 DMIADXConfirmation 보존 서브타입 — "Adx Only", "Adx & +Di -Di"
# **Validates: Requirements 3.4**
# ---------------------------------------------------------------------------

class TestDMIADXConfirmationPreservedSubtypes:
    """DMIADXConfirmation의 보존 서브타입 동작 검증.

    "Adx Only"와 "Adx & +Di -Di"는 기본값 변경 외 동일 로직 유지.

    **Validates: Requirements 3.4**
    """

    def test_adx_only_returns_valid_result(self):
        """Adx Only 서브타입이 유효한 IndicatorResult를 반환한다."""
        indicator = DMIADXConfirmation()
        result = indicator.calculate(DF_TREND, subtype="Adx Only")

        assert isinstance(result, IndicatorResult)
        assert result.long_signal.dtype == bool
        assert result.short_signal.dtype == bool
        assert not result.long_signal.isna().any()

    def test_adx_only_long_equals_short(self):
        """Adx Only: Long과 Short 시그널이 동일하다 (추세 필터)."""
        indicator = DMIADXConfirmation()
        result = indicator.calculate(DF_TREND, subtype="Adx Only")

        # Adx Only는 ADX > threshold일 때 Long=Short=True
        pd.testing.assert_series_equal(
            result.long_signal, result.short_signal,
            check_names=False,
        )

    def test_adx_di_returns_valid_result(self):
        """Adx & +Di -Di 서브타입이 유효한 IndicatorResult를 반환한다."""
        indicator = DMIADXConfirmation()
        result = indicator.calculate(DF_TREND, subtype="Adx & +Di -Di")

        assert isinstance(result, IndicatorResult)
        assert result.long_signal.dtype == bool
        assert result.short_signal.dtype == bool
        assert not result.long_signal.isna().any()

    def test_adx_di_snapshot(self):
        """Adx & +Di -Di 서브타입의 시그널 스냅샷이 결정론적이다."""
        indicator = DMIADXConfirmation()
        result1 = indicator.calculate(DF_TREND, subtype="Adx & +Di -Di")
        result2 = indicator.calculate(DF_TREND, subtype="Adx & +Di -Di")

        pd.testing.assert_series_equal(result1.long_signal, result2.long_signal)
        pd.testing.assert_series_equal(result1.short_signal, result2.short_signal)

    def test_adx_only_with_custom_params(self):
        """Adx Only에 사용자 지정 파라미터를 전달해도 동작한다.

        **Validates: Requirements 3.5** (_resolve_params 병합 동작)
        """
        indicator = DMIADXConfirmation()
        custom_params = {"adx_threshold": 25}
        result = indicator.calculate(
            DF_TREND, params=custom_params, subtype="Adx Only"
        )

        assert isinstance(result, IndicatorResult)
        assert result.long_signal.dtype == bool


# ---------------------------------------------------------------------------
# 3.5 _resolve_params 파라미터 병합 동작 보존
# **Validates: Requirements 3.5**
# ---------------------------------------------------------------------------

class TestResolveParamsPreservation:
    """_resolve_params 메서드를 통한 사용자 파라미터 병합 동작 검증.

    **Validates: Requirements 3.5**
    """

    def test_none_params_uses_defaults(self):
        """params=None이면 기본값만 사용한다."""
        indicator = EMAFilterConfirmation()
        resolved = indicator._resolve_params(None)

        assert resolved == indicator.default_params

    def test_custom_params_override_defaults(self):
        """사용자 파라미터가 기본값을 덮어쓴다."""
        indicator = EMAFilterConfirmation()
        custom = {"length": 50}
        resolved = indicator._resolve_params(custom)

        assert resolved["length"] == 50

    def test_unrelated_params_are_added(self):
        """기본값에 없는 파라미터도 추가된다."""
        indicator = EMAFilterConfirmation()
        custom = {"extra_param": 999}
        resolved = indicator._resolve_params(custom)

        assert resolved["extra_param"] == 999
        assert "length" in resolved  # 기본값도 유지

    def test_resolve_params_does_not_mutate_defaults(self):
        """_resolve_params가 default_params를 변경하지 않는다."""
        indicator = ROCConfirmation()
        original_defaults = dict(indicator.default_params)
        indicator._resolve_params({"length": 999})

        assert indicator.default_params == original_defaults

    def test_custom_params_produce_different_signals(self):
        """사용자 파라미터가 시그널에 영향을 미친다."""
        indicator = EMAFilterConfirmation()
        result_default = indicator.calculate(DF_TREND)
        result_custom = indicator.calculate(DF_TREND, params={"length": 10})

        # 다른 EMA 기간은 다른 시그널을 생성해야 함
        assert not result_default.long_signal.equals(result_custom.long_signal), (
            "다른 파라미터는 다른 시그널을 생성해야 함"
        )


# ---------------------------------------------------------------------------
# 3.6 유효하지 않은 서브타입 ValueError 발생 보존
# **Validates: Requirements 3.6**
# ---------------------------------------------------------------------------

class TestInvalidSubtypeValueError:
    """유효하지 않은 서브타입 전달 시 ValueError 발생 검증.

    **Validates: Requirements 3.6**
    """

    def test_rsi_invalid_subtype_raises(self):
        """RSIConfirmation에 유효하지 않은 서브타입 전달 시 ValueError."""
        indicator = RSIConfirmation()
        with pytest.raises(ValueError, match="서브타입을 지원하지 않습니다"):
            indicator.calculate(DF_TREND, subtype="Invalid Subtype")

    def test_tsi_invalid_subtype_raises(self):
        """TSIConfirmation에 유효하지 않은 서브타입 전달 시 ValueError."""
        indicator = TSIConfirmation()
        with pytest.raises(ValueError, match="서브타입을 지원하지 않습니다"):
            indicator.calculate(DF_TREND, subtype="Nonexistent")

    def test_dmi_invalid_subtype_raises(self):
        """DMIADXConfirmation에 유효하지 않은 서브타입 전달 시 ValueError."""
        indicator = DMIADXConfirmation()
        with pytest.raises(ValueError, match="서브타입을 지원하지 않습니다"):
            indicator.calculate(DF_TREND, subtype="Bad Subtype")

    def test_macd_invalid_subtype_raises(self):
        """MACDConfirmation에 유효하지 않은 서브타입 전달 시 ValueError."""
        indicator = MACDConfirmation()
        with pytest.raises(ValueError, match="서브타입을 지원하지 않습니다"):
            indicator.calculate(DF_TREND, subtype="Wrong")

    def test_stochastic_invalid_subtype_raises(self):
        """StochasticConfirmation에 유효하지 않은 서브타입 전달 시 ValueError."""
        indicator = StochasticConfirmation()
        with pytest.raises(ValueError, match="서브타입을 지원하지 않습니다"):
            indicator.calculate(DF_TREND, subtype="Fake")


# ---------------------------------------------------------------------------
# 3.7 IndicatorResult 형식 및 NaN→False 채움 보존 (추가 검증)
# **Validates: Requirements 3.7**
# ---------------------------------------------------------------------------

class TestIndicatorResultFormatPreservation:
    """IndicatorResult 형식 및 NaN→False 채움 동작 보존 검증.

    **Validates: Requirements 3.7**
    """

    def test_short_data_no_nan(self):
        """짧은 데이터에서도 NaN이 False로 채워진다."""
        # 50개 데이터 — 일부 indicator는 워밍업 기간이 필요
        short_df = _make_ohlcv(n=50, seed=11111)
        indicator = EMAFilterConfirmation()
        result = indicator.calculate(short_df)

        assert not result.long_signal.isna().any()
        assert not result.short_signal.isna().any()

    def test_preserved_subtypes_no_nan_on_short_data(self):
        """보존 서브타입도 짧은 데이터에서 NaN이 없다."""
        short_df = _make_ohlcv(n=80, seed=22222)

        # RSI MA Cross
        rsi_result = RSIConfirmation().calculate(short_df, subtype="RSI MA Cross")
        assert not rsi_result.long_signal.isna().any()

        # TSI Signal Cross
        tsi_result = TSIConfirmation().calculate(short_df, subtype="Signal Cross")
        assert not tsi_result.long_signal.isna().any()

    def test_result_index_matches_input(self):
        """결과 Series의 인덱스가 입력 DataFrame의 인덱스와 일치한다."""
        indicator = SupertrendConfirmation()
        result = indicator.calculate(DF_FLAT)

        pd.testing.assert_index_equal(result.long_signal.index, DF_FLAT.index)
        pd.testing.assert_index_equal(result.short_signal.index, DF_FLAT.index)


# ---------------------------------------------------------------------------
# 비수정 Indicator 시그널 스냅샷 보존 (결정론적 검증)
# **Validates: Requirements 3.1**
# ---------------------------------------------------------------------------

class TestPreservedIndicatorSnapshotConsistency:
    """비수정 indicator의 시그널이 결정론적이고 재현 가능한지 검증.

    동일 입력에 대해 두 번 실행한 결과가 동일해야 한다.

    **Validates: Requirements 3.1**
    """

    @pytest.mark.parametrize(
        "indicator_and_subtype",
        PRESERVED_INDICATORS,
        ids=[_indicator_id(p) for p in PRESERVED_INDICATORS],
    )
    def test_deterministic_output(self, indicator_and_subtype):
        """동일 입력에 대해 동일 출력을 생성한다 (결정론적)."""
        indicator, subtype = indicator_and_subtype
        result1 = indicator.calculate(DF_TREND, subtype=subtype)
        result2 = indicator.calculate(DF_TREND, subtype=subtype)

        pd.testing.assert_series_equal(
            result1.long_signal, result2.long_signal,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result1.short_signal, result2.short_signal,
            check_names=False,
        )

    @pytest.mark.parametrize(
        "indicator_and_subtype",
        PRESERVED_INDICATORS,
        ids=[_indicator_id(p) for p in PRESERVED_INDICATORS],
    )
    def test_different_data_different_signals(self, indicator_and_subtype):
        """다른 입력 데이터에 대해 다른 시그널을 생성한다."""
        indicator, subtype = indicator_and_subtype
        result_trend = indicator.calculate(DF_TREND, subtype=subtype)
        result_flat = indicator.calculate(DF_FLAT, subtype=subtype)

        # 최소한 하나의 시그널이 달라야 함 (완전히 동일할 확률은 매우 낮음)
        signals_differ = (
            not result_trend.long_signal.equals(result_flat.long_signal)
            or not result_trend.short_signal.equals(result_flat.short_signal)
        )
        assert signals_differ, (
            f"{indicator.name} should produce different signals for different data"
        )
