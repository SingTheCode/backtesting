"""지표 기본 인터페이스 단위 테스트.

IndicatorResult, BaseIndicator, ConfirmationIndicator의
구조, 계약, 에러 처리를 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.indicators.base import (
    BaseIndicator,
    ConfirmationIndicator,
    IndicatorResult,
)


# ---------------------------------------------------------------------------
# 테스트용 구체 클래스
# ---------------------------------------------------------------------------

class DummyLeadingIndicator(BaseIndicator):
    """테스트용 리딩 지표 구현체."""

    @property
    def name(self) -> str:
        return "DummyLeading"

    @property
    def default_params(self) -> dict:
        return {"length": 14, "threshold": 0.5}

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        p = self._resolve_params(params)
        length = p["length"]
        long_signal = df["Close"] > df["Close"].shift(length)
        short_signal = df["Close"] < df["Close"].shift(length)
        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DummyConfirmation(ConfirmationIndicator):
    """테스트용 확인 지표 구현체 (서브타입 없음)."""

    @property
    def name(self) -> str:
        return "DummyConfirmation"

    @property
    def default_params(self) -> dict:
        return {"period": 20}

    def _calculate_impl(
        self, df: pd.DataFrame, params: dict, subtype: str | None
    ) -> IndicatorResult:
        n = len(df)
        return IndicatorResult(
            long_signal=pd.Series([True] * n, index=df.index),
            short_signal=pd.Series([False] * n, index=df.index),
        )


class DummySubtypeConfirmation(ConfirmationIndicator):
    """테스트용 확인 지표 구현체 (서브타입 있음)."""

    @property
    def name(self) -> str:
        return "DummySubtype"

    @property
    def default_params(self) -> dict:
        return {"length": 14}

    @property
    def subtypes(self) -> list[str]:
        return ["Signal Cross", "Zero line cross"]

    def _calculate_impl(
        self, df: pd.DataFrame, params: dict, subtype: str | None
    ) -> IndicatorResult:
        n = len(df)
        if subtype == "Signal Cross":
            return IndicatorResult(
                long_signal=pd.Series([True] * n, index=df.index),
                short_signal=pd.Series([False] * n, index=df.index),
            )
        return IndicatorResult(
            long_signal=pd.Series([False] * n, index=df.index),
            short_signal=pd.Series([True] * n, index=df.index),
        )


# ---------------------------------------------------------------------------
# 테스트 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """50행짜리 샘플 OHLCV DataFrame을 생성한다."""
    np.random.seed(42)
    n = 50
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close - np.random.rand(n) * 0.3,
            "High": close + np.random.rand(n) * 0.5,
            "Low": close - np.random.rand(n) * 0.5,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, size=n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# IndicatorResult 테스트
# ---------------------------------------------------------------------------

class TestIndicatorResult:
    """IndicatorResult dataclass 검증."""

    def test_create_with_series(self):
        """pd.Series로 정상 생성된다."""
        long_s = pd.Series([True, False, True])
        short_s = pd.Series([False, True, False])
        result = IndicatorResult(long_signal=long_s, short_signal=short_s)
        assert isinstance(result.long_signal, pd.Series)
        assert isinstance(result.short_signal, pd.Series)
        assert len(result.long_signal) == 3
        assert len(result.short_signal) == 3

    def test_fields_are_accessible(self):
        """long_signal, short_signal 필드에 직접 접근 가능하다."""
        result = IndicatorResult(
            long_signal=pd.Series([True]),
            short_signal=pd.Series([False]),
        )
        assert result.long_signal.iloc[0] is True or result.long_signal.iloc[0] == True
        assert result.short_signal.iloc[0] is False or result.short_signal.iloc[0] == False


# ---------------------------------------------------------------------------
# BaseIndicator 테스트
# ---------------------------------------------------------------------------

class TestBaseIndicator:
    """BaseIndicator 추상 클래스 검증."""

    def test_cannot_instantiate_directly(self):
        """추상 클래스를 직접 인스턴스화할 수 없다."""
        with pytest.raises(TypeError):
            BaseIndicator()  # type: ignore[abstract]

    def test_concrete_class_name(self):
        """구체 클래스의 name 프로퍼티가 올바르게 반환된다."""
        indicator = DummyLeadingIndicator()
        assert indicator.name == "DummyLeading"

    def test_concrete_class_default_params(self):
        """구체 클래스의 default_params가 올바르게 반환된다."""
        indicator = DummyLeadingIndicator()
        assert indicator.default_params == {"length": 14, "threshold": 0.5}

    def test_calculate_returns_indicator_result(self, sample_ohlcv: pd.DataFrame):
        """calculate()가 IndicatorResult를 반환한다."""
        indicator = DummyLeadingIndicator()
        result = indicator.calculate(sample_ohlcv)
        assert isinstance(result, IndicatorResult)

    def test_calculate_result_length_matches_input(self, sample_ohlcv: pd.DataFrame):
        """반환된 시그널 길이가 입력 DataFrame 행 수와 동일하다."""
        indicator = DummyLeadingIndicator()
        result = indicator.calculate(sample_ohlcv)
        assert len(result.long_signal) == len(sample_ohlcv)
        assert len(result.short_signal) == len(sample_ohlcv)

    def test_calculate_with_custom_params(self, sample_ohlcv: pd.DataFrame):
        """사용자 지정 파라미터가 기본값을 덮어쓴다."""
        indicator = DummyLeadingIndicator()
        result = indicator.calculate(sample_ohlcv, params={"length": 5})
        assert isinstance(result, IndicatorResult)
        assert len(result.long_signal) == len(sample_ohlcv)

    def test_resolve_params_merges_defaults(self):
        """_resolve_params가 기본값과 사용자 값을 올바르게 병합한다."""
        indicator = DummyLeadingIndicator()
        resolved = indicator._resolve_params({"length": 7})
        assert resolved == {"length": 7, "threshold": 0.5}

    def test_resolve_params_none_returns_defaults(self):
        """_resolve_params에 None 전달 시 기본값만 반환한다."""
        indicator = DummyLeadingIndicator()
        resolved = indicator._resolve_params(None)
        assert resolved == {"length": 14, "threshold": 0.5}

    def test_signals_are_boolean_dtype(self, sample_ohlcv: pd.DataFrame):
        """반환된 시그널이 boolean 값을 포함한다."""
        indicator = DummyLeadingIndicator()
        result = indicator.calculate(sample_ohlcv)
        assert result.long_signal.dtype == bool
        assert result.short_signal.dtype == bool


# ---------------------------------------------------------------------------
# ConfirmationIndicator 테스트
# ---------------------------------------------------------------------------

class TestConfirmationIndicator:
    """ConfirmationIndicator 서브타입 지원 검증."""

    def test_cannot_instantiate_directly(self):
        """추상 클래스를 직접 인스턴스화할 수 없다."""
        with pytest.raises(TypeError):
            ConfirmationIndicator()  # type: ignore[abstract]

    def test_default_subtypes_empty(self):
        """서브타입이 없는 확인 지표는 빈 리스트를 반환한다."""
        indicator = DummyConfirmation()
        assert indicator.subtypes == []

    def test_subtypes_returns_list(self):
        """서브타입이 있는 확인 지표는 문자열 리스트를 반환한다."""
        indicator = DummySubtypeConfirmation()
        assert indicator.subtypes == ["Signal Cross", "Zero line cross"]

    def test_calculate_without_subtype(self, sample_ohlcv: pd.DataFrame):
        """서브타입 없이 호출하면 정상 동작한다."""
        indicator = DummyConfirmation()
        result = indicator.calculate(sample_ohlcv)
        assert isinstance(result, IndicatorResult)
        assert len(result.long_signal) == len(sample_ohlcv)

    def test_calculate_with_valid_subtype(self, sample_ohlcv: pd.DataFrame):
        """유효한 서브타입으로 호출하면 해당 로직이 실행된다."""
        indicator = DummySubtypeConfirmation()
        result = indicator.calculate(sample_ohlcv, subtype="Signal Cross")
        assert isinstance(result, IndicatorResult)
        assert result.long_signal.all()

    def test_calculate_with_different_subtype(self, sample_ohlcv: pd.DataFrame):
        """다른 서브타입은 다른 결과를 반환한다."""
        indicator = DummySubtypeConfirmation()
        result = indicator.calculate(sample_ohlcv, subtype="Zero line cross")
        assert isinstance(result, IndicatorResult)
        assert result.short_signal.all()

    def test_calculate_with_invalid_subtype_raises(self, sample_ohlcv: pd.DataFrame):
        """지원하지 않는 서브타입 전달 시 ValueError가 발생한다."""
        indicator = DummySubtypeConfirmation()
        with pytest.raises(ValueError, match="지원하지 않습니다"):
            indicator.calculate(sample_ohlcv, subtype="InvalidType")

    def test_calculate_with_params_and_subtype(self, sample_ohlcv: pd.DataFrame):
        """파라미터와 서브타입을 동시에 전달할 수 있다."""
        indicator = DummySubtypeConfirmation()
        result = indicator.calculate(
            sample_ohlcv, params={"length": 7}, subtype="Signal Cross"
        )
        assert isinstance(result, IndicatorResult)

    def test_no_subtype_indicator_ignores_none_subtype(
        self, sample_ohlcv: pd.DataFrame
    ):
        """서브타입이 없는 지표에 subtype=None 전달 시 정상 동작한다."""
        indicator = DummyConfirmation()
        result = indicator.calculate(sample_ohlcv, subtype=None)
        assert isinstance(result, IndicatorResult)
