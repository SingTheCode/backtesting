"""IndicatorCache 단위 테스트 모듈.

캐시 히트/미스, 멱등성, 통계, clear() 등 핵심 동작을 검증한다.
"""

import pandas as pd
import numpy as np

from sp500_backtest.engine.cache import IndicatorCache
from sp500_backtest.indicators.base import IndicatorResult


def _make_df(n: int = 50) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame 생성.

    Args:
        n: 행 수.

    Returns:
        OHLCV DataFrame.
    """
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame({
        "Open": close + rng.uniform(-1, 1, n),
        "High": close + abs(rng.standard_normal(n)),
        "Low": close - abs(rng.standard_normal(n)),
        "Close": close,
        "Volume": rng.integers(1000, 10000, n),
    }, index=pd.date_range("2022-01-01", periods=n, freq="B"))


def _dummy_compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
    """더미 지표 계산 함수.

    Args:
        df: OHLCV DataFrame.
        params: 파라미터 딕셔너리.

    Returns:
        IndicatorResult: 모두 False인 시그널.
    """
    length = len(df)
    return IndicatorResult(
        long_signal=pd.Series([False] * length, index=df.index),
        short_signal=pd.Series([False] * length, index=df.index),
    )


class TestIndicatorCacheHitMiss:
    """캐시 히트/미스 동작 검증."""

    def test_cache_miss_calls_compute(self):
        """캐시 미스 시 compute_fn이 호출되어 결과를 반환한다."""
        cache = IndicatorCache()
        df = _make_df()
        call_count = 0

        def counting_compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
            nonlocal call_count
            call_count += 1
            return _dummy_compute(df, params)

        result = cache.get_or_compute("RSI", {"length": 14}, df, counting_compute)

        assert call_count == 1
        assert isinstance(result, IndicatorResult)
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_hit_skips_compute(self):
        """캐시 히트 시 compute_fn이 호출되지 않고 캐시된 결과를 반환한다."""
        cache = IndicatorCache()
        df = _make_df()
        call_count = 0

        def counting_compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
            nonlocal call_count
            call_count += 1
            return _dummy_compute(df, params)

        params = {"length": 14}
        first = cache.get_or_compute("RSI", params, df, counting_compute)
        second = cache.get_or_compute("RSI", params, df, counting_compute)

        assert call_count == 1  # 두 번째 호출에서 compute_fn 미호출
        assert first is second  # 동일 객체 반환
        assert cache.hits == 1
        assert cache.misses == 1

    def test_different_params_trigger_recompute(self):
        """다른 파라미터는 별도 캐시 엔트리로 계산된다."""
        cache = IndicatorCache()
        df = _make_df()

        cache.get_or_compute("RSI", {"length": 14}, df, _dummy_compute)
        cache.get_or_compute("RSI", {"length": 21}, df, _dummy_compute)

        assert cache.misses == 2
        assert cache.size == 2

    def test_different_indicator_names_trigger_recompute(self):
        """다른 지표명은 별도 캐시 엔트리로 계산된다."""
        cache = IndicatorCache()
        df = _make_df()
        params = {"length": 14}

        cache.get_or_compute("RSI", params, df, _dummy_compute)
        cache.get_or_compute("CCI", params, df, _dummy_compute)

        assert cache.misses == 2
        assert cache.size == 2


class TestIndicatorCacheIdempotency:
    """캐시 멱등성 검증 — 동일 키로 두 번 호출 시 동일 결과 반환."""

    def test_two_calls_return_identical_results(self):
        """동일 파라미터로 두 번 호출하면 동일한 IndicatorResult를 반환한다."""
        cache = IndicatorCache()
        df = _make_df()
        params = {"fast": 12, "slow": 26, "signal": 9}

        r1 = cache.get_or_compute("MACD", params, df, _dummy_compute)
        r2 = cache.get_or_compute("MACD", params, df, _dummy_compute)

        assert r1 is r2
        pd.testing.assert_series_equal(r1.long_signal, r2.long_signal)
        pd.testing.assert_series_equal(r1.short_signal, r2.short_signal)


class TestIndicatorCacheStats:
    """캐시 통계(hits, misses, size) 검증."""

    def test_initial_stats_are_zero(self):
        """초기 상태에서 모든 통계는 0이다."""
        cache = IndicatorCache()
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.size == 0

    def test_stats_accumulate_correctly(self):
        """여러 호출 후 통계가 정확히 누적된다."""
        cache = IndicatorCache()
        df = _make_df()

        # 3개 서로 다른 키 → 3 misses
        cache.get_or_compute("A", {"x": 1}, df, _dummy_compute)
        cache.get_or_compute("B", {"x": 2}, df, _dummy_compute)
        cache.get_or_compute("C", {"x": 3}, df, _dummy_compute)

        # 기존 키 재사용 → 2 hits
        cache.get_or_compute("A", {"x": 1}, df, _dummy_compute)
        cache.get_or_compute("B", {"x": 2}, df, _dummy_compute)

        assert cache.misses == 3
        assert cache.hits == 2
        assert cache.size == 3


class TestIndicatorCacheClear:
    """clear() 메서드 검증."""

    def test_clear_resets_cache_and_stats(self):
        """clear() 호출 후 캐시와 통계가 모두 초기화된다."""
        cache = IndicatorCache()
        df = _make_df()

        cache.get_or_compute("RSI", {"length": 14}, df, _dummy_compute)
        cache.get_or_compute("RSI", {"length": 14}, df, _dummy_compute)

        assert cache.size == 1
        assert cache.hits == 1

        cache.clear()

        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_compute_after_clear(self):
        """clear() 후 동일 키로 호출하면 다시 계산한다."""
        cache = IndicatorCache()
        df = _make_df()
        call_count = 0

        def counting_compute(df: pd.DataFrame, params: dict) -> IndicatorResult:
            nonlocal call_count
            call_count += 1
            return _dummy_compute(df, params)

        params = {"length": 14}
        cache.get_or_compute("RSI", params, df, counting_compute)
        cache.clear()
        cache.get_or_compute("RSI", params, df, counting_compute)

        assert call_count == 2  # clear 후 재계산


class TestIndicatorCacheKeyGeneration:
    """frozenset 기반 캐시 키 생성 검증."""

    def test_param_order_does_not_matter(self):
        """파라미터 딕셔너리 순서가 달라도 동일 키로 인식한다."""
        cache = IndicatorCache()
        df = _make_df()

        params_a = {"fast": 12, "slow": 26}
        params_b = {"slow": 26, "fast": 12}

        r1 = cache.get_or_compute("MACD", params_a, df, _dummy_compute)
        r2 = cache.get_or_compute("MACD", params_b, df, _dummy_compute)

        assert r1 is r2
        assert cache.hits == 1
        assert cache.misses == 1

    def test_empty_params(self):
        """빈 파라미터 딕셔너리도 정상 동작한다."""
        cache = IndicatorCache()
        df = _make_df()

        result = cache.get_or_compute("VWAP", {}, df, _dummy_compute)
        assert isinstance(result, IndicatorResult)
        assert cache.size == 1
