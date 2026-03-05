"""지표 공통 유틸리티 함수 단위 테스트.

ma(), atr(), crossover(), crossunder(), stoch(), rsi(), true_range()의
정확성, 엣지 케이스, NaN 처리를 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.indicators.utils import (
    atr,
    crossover,
    crossunder,
    ma,
    rsi,
    stoch,
    true_range,
)


# ---------------------------------------------------------------------------
# 테스트 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_series() -> pd.Series:
    """20개 값을 가진 샘플 시계열을 생성한다."""
    return pd.Series(
        [10, 11, 12, 11, 13, 14, 13, 15, 16, 14, 12, 13, 15, 17, 16, 18, 19, 17, 20, 21],
        dtype=float,
    )


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """30행짜리 샘플 OHLCV DataFrame을 생성한다."""
    np.random.seed(42)
    n = 30
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# ma() 테스트
# ---------------------------------------------------------------------------


class TestMA:
    """이동평균 함수 검증."""

    def test_sma_basic(self, sample_series: pd.Series):
        """SMA가 올바른 단순 평균을 반환한다."""
        result = ma(sample_series, length=3, ma_type="SMA")
        assert len(result) == len(sample_series)
        # 처음 2개는 NaN (min_periods=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # 3번째 값: (10+11+12)/3 = 11.0
        assert result.iloc[2] == pytest.approx(11.0)

    def test_ema_returns_series(self, sample_series: pd.Series):
        """EMA가 입력과 동일한 길이의 Series를 반환한다."""
        result = ma(sample_series, length=5, ma_type="EMA")
        assert len(result) == len(sample_series)
        # EMA는 첫 값부터 계산 가능 (adjust=False)
        assert not pd.isna(result.iloc[0])

    def test_rma_returns_series(self, sample_series: pd.Series):
        """RMA(Wilder's smoothing)가 Series를 반환한다."""
        result = ma(sample_series, length=5, ma_type="RMA")
        assert len(result) == len(sample_series)
        assert not pd.isna(result.iloc[0])

    def test_wma_basic(self, sample_series: pd.Series):
        """WMA가 가중 평균을 올바르게 계산한다."""
        result = ma(sample_series, length=3, ma_type="WMA")
        assert len(result) == len(sample_series)
        # 처음 2개는 NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # 3번째: (10*1 + 11*2 + 12*3) / (1+2+3) = 68/6 ≈ 11.333
        expected = (10 * 1 + 11 * 2 + 12 * 3) / 6
        assert result.iloc[2] == pytest.approx(expected)

    def test_hma_returns_series(self, sample_series: pd.Series):
        """HMA가 입력과 동일한 길이의 Series를 반환한다."""
        result = ma(sample_series, length=9, ma_type="HMA")
        assert len(result) == len(sample_series)

    def test_vwma_basic(self, sample_ohlcv: pd.DataFrame):
        """VWMA가 거래량 가중 평균을 반환한다."""
        result = ma(
            sample_ohlcv["Close"],
            length=5,
            ma_type="VWMA",
            volume=sample_ohlcv["Volume"],
        )
        assert len(result) == len(sample_ohlcv)
        # 처음 4개는 NaN
        assert pd.isna(result.iloc[3])
        assert not pd.isna(result.iloc[4])

    def test_vwma_without_volume_raises(self, sample_series: pd.Series):
        """VWMA 선택 시 volume 미전달하면 ValueError가 발생한다."""
        with pytest.raises(ValueError, match="volume 데이터가 필요"):
            ma(sample_series, length=5, ma_type="VWMA")

    def test_invalid_ma_type_raises(self, sample_series: pd.Series):
        """지원하지 않는 ma_type 전달 시 ValueError가 발생한다."""
        with pytest.raises(ValueError, match="지원하지 않는 이동평균 유형"):
            ma(sample_series, length=5, ma_type="INVALID")

    def test_ma_type_case_insensitive(self, sample_series: pd.Series):
        """ma_type은 대소문자를 구분하지 않는다."""
        result_lower = ma(sample_series, length=5, ma_type="sma")
        result_upper = ma(sample_series, length=5, ma_type="SMA")
        pd.testing.assert_series_equal(result_lower, result_upper)

    def test_sma_all_same_values(self):
        """모든 값이 동일하면 SMA도 동일한 값을 반환한다."""
        s = pd.Series([5.0] * 10)
        result = ma(s, length=3, ma_type="SMA")
        # NaN이 아닌 값은 모두 5.0
        valid = result.dropna()
        assert (valid == 5.0).all()


# ---------------------------------------------------------------------------
# true_range() 테스트
# ---------------------------------------------------------------------------


class TestTrueRange:
    """True Range 함수 검증."""

    def test_returns_series(self, sample_ohlcv: pd.DataFrame):
        """true_range()가 입력과 동일한 길이의 Series를 반환한다."""
        result = true_range(sample_ohlcv)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_first_row_is_high_minus_low(self, sample_ohlcv: pd.DataFrame):
        """첫 번째 행의 TR은 High - Low이다 (전일 종가 없음)."""
        result = true_range(sample_ohlcv)
        expected = sample_ohlcv["High"].iloc[0] - sample_ohlcv["Low"].iloc[0]
        assert result.iloc[0] == pytest.approx(expected)

    def test_all_values_non_negative(self, sample_ohlcv: pd.DataFrame):
        """True Range는 항상 0 이상이다."""
        result = true_range(sample_ohlcv)
        assert (result.dropna() >= 0).all()

    def test_manual_calculation(self):
        """수동 계산과 결과가 일치한다."""
        df = pd.DataFrame(
            {
                "High": [12.0, 15.0, 14.0],
                "Low": [10.0, 11.0, 12.0],
                "Close": [11.0, 14.0, 13.0],
            }
        )
        result = true_range(df)
        # 행 0: High-Low = 2.0 (전일 종가 없음, tr2/tr3는 NaN이므로 max는 2.0)
        assert result.iloc[0] == pytest.approx(2.0)
        # 행 1: max(15-11, |15-11|, |11-11|) = max(4, 4, 0) = 4.0
        assert result.iloc[1] == pytest.approx(4.0)
        # 행 2: max(14-12, |14-14|, |12-14|) = max(2, 0, 2) = 2.0
        assert result.iloc[2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# atr() 테스트
# ---------------------------------------------------------------------------


class TestATR:
    """Average True Range 함수 검증."""

    def test_returns_series(self, sample_ohlcv: pd.DataFrame):
        """atr()이 입력과 동일한 길이의 Series를 반환한다."""
        result = atr(sample_ohlcv, length=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_all_values_non_negative(self, sample_ohlcv: pd.DataFrame):
        """ATR은 항상 0 이상이다."""
        result = atr(sample_ohlcv, length=5)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr_smooths_true_range(self, sample_ohlcv: pd.DataFrame):
        """ATR은 True Range의 평활화 버전이다."""
        tr = true_range(sample_ohlcv)
        atr_result = atr(sample_ohlcv, length=5)
        # ATR의 표준편차가 TR의 표준편차보다 작거나 같아야 함 (평활화 효과)
        assert atr_result.dropna().std() <= tr.dropna().std()


# ---------------------------------------------------------------------------
# crossover() / crossunder() 테스트
# ---------------------------------------------------------------------------


class TestCrossover:
    """크로스오버 감지 함수 검증."""

    def test_basic_crossover(self):
        """A가 B를 상향 돌파하는 시점을 감지한다."""
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        b = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
        result = crossover(a, b)
        # 인덱스 3에서 a(4) > b(3) 이고 이전 a(3) <= b(3)
        assert result.iloc[3] is True or result.iloc[3] == True
        # 인덱스 0, 1, 2에서는 False
        assert result.iloc[0] == False
        assert result.iloc[1] == False
        assert result.iloc[2] == False

    def test_no_crossover_when_always_above(self):
        """A가 항상 B 위에 있으면 크로스오버가 발생하지 않는다."""
        a = pd.Series([10.0, 11.0, 12.0, 13.0])
        b = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = crossover(a, b)
        assert not result.any()

    def test_crossover_returns_boolean(self):
        """반환값이 boolean dtype이다."""
        a = pd.Series([1.0, 3.0])
        b = pd.Series([2.0, 2.0])
        result = crossover(a, b)
        assert result.dtype == bool


class TestCrossunder:
    """크로스언더 감지 함수 검증."""

    def test_basic_crossunder(self):
        """A가 B를 하향 돌파하는 시점을 감지한다."""
        a = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        b = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
        result = crossunder(a, b)
        # 인덱스 2에서 a(3) < b(3)이 아니므로 False, 인덱스 3에서 a(2) < b(3) 이고 이전 a(3) >= b(3)
        assert result.iloc[3] is True or result.iloc[3] == True

    def test_no_crossunder_when_always_below(self):
        """A가 항상 B 아래에 있으면 크로스언더가 발생하지 않는다."""
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([10.0, 11.0, 12.0, 13.0])
        result = crossunder(a, b)
        assert not result.any()

    def test_crossunder_returns_boolean(self):
        """반환값이 boolean dtype이다."""
        a = pd.Series([3.0, 1.0])
        b = pd.Series([2.0, 2.0])
        result = crossunder(a, b)
        assert result.dtype == bool

    def test_crossover_crossunder_mutually_exclusive(self):
        """동일 시점에서 크로스오버와 크로스언더가 동시에 발생하지 않는다."""
        a = pd.Series([1.0, 3.0, 1.0, 3.0, 1.0])
        b = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
        co = crossover(a, b)
        cu = crossunder(a, b)
        # 동일 시점에서 둘 다 True인 경우가 없어야 함
        assert not (co & cu).any()


# ---------------------------------------------------------------------------
# stoch() 테스트
# ---------------------------------------------------------------------------


class TestStoch:
    """Stochastic %K 함수 검증."""

    def test_returns_series(self, sample_ohlcv: pd.DataFrame):
        """stoch()이 입력과 동일한 길이의 Series를 반환한다."""
        result = stoch(
            sample_ohlcv["Close"],
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            length=14,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)

    def test_range_0_to_100(self, sample_ohlcv: pd.DataFrame):
        """%K 값이 0~100 범위 내에 있다."""
        result = stoch(
            sample_ohlcv["Close"],
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            length=5,
        )
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_manual_calculation(self):
        """수동 계산과 결과가 일치한다."""
        close = pd.Series([10.0, 12.0, 11.0, 14.0, 13.0])
        high = pd.Series([11.0, 13.0, 12.0, 15.0, 14.0])
        low = pd.Series([9.0, 11.0, 10.0, 13.0, 12.0])
        result = stoch(close, high, low, length=3)
        # 인덱스 2: lowest_low=9, highest_high=13, %K = 100*(11-9)/(13-9) = 50
        assert result.iloc[2] == pytest.approx(50.0)

    def test_nan_at_beginning(self):
        """기간 미달 구간에서 NaN을 반환한다."""
        close = pd.Series([10.0, 12.0, 11.0])
        high = pd.Series([11.0, 13.0, 12.0])
        low = pd.Series([9.0, 11.0, 10.0])
        result = stoch(close, high, low, length=5)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# rsi() 테스트
# ---------------------------------------------------------------------------


class TestRSI:
    """RSI 함수 검증."""

    def test_returns_series(self, sample_series: pd.Series):
        """rsi()가 입력과 동일한 길이의 Series를 반환한다."""
        result = rsi(sample_series, length=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)

    def test_range_0_to_100(self, sample_series: pd.Series):
        """RSI 값이 0~100 범위 내에 있다."""
        result = rsi(sample_series, length=5)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_monotonic_increase_gives_high_rsi(self):
        """단조 증가 시계열은 높은 RSI를 반환한다."""
        s = pd.Series(range(1, 21), dtype=float)
        result = rsi(s, length=14)
        # 마지막 값은 100에 가까워야 함
        assert result.iloc[-1] > 90

    def test_monotonic_decrease_gives_low_rsi(self):
        """단조 감소 시계열은 낮은 RSI를 반환한다."""
        s = pd.Series(range(20, 0, -1), dtype=float)
        result = rsi(s, length=14)
        # 마지막 값은 0에 가까워야 함
        assert result.iloc[-1] < 10

    def test_first_value_is_nan(self):
        """첫 번째 값(diff 불가)은 NaN이다."""
        s = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0])
        result = rsi(s, length=3)
        assert pd.isna(result.iloc[0])
