"""지표 계산 결과 캐싱 모듈.

동일 지표가 여러 조합에서 사용될 때 중복 계산을 방지하기 위한
인메모리 캐시를 제공한다.
"""

import threading
from typing import Callable

import pandas as pd

from sp500_backtest.indicators.base import IndicatorResult


class IndicatorCache:
    """지표 계산 결과를 캐싱하는 스레드 안전 인메모리 캐시.

    캐시 키는 (indicator_name, frozenset(params.items())) 튜플로 구성되며,
    동일 지표+파라미터 조합의 중복 계산을 방지한다.
    """

    def __init__(self) -> None:
        """캐시 초기화."""
        self._cache: dict[tuple[str, frozenset], IndicatorResult] = {}
        """캐시 저장소 (키: (지표명, 파라미터 frozenset), 값: IndicatorResult)"""
        self._hits: int = 0
        """캐시 히트 횟수"""
        self._misses: int = 0
        """캐시 미스 횟수"""
        self._lock: threading.Lock = threading.Lock()
        """스레드 안전을 위한 락"""

    def _make_key(self, indicator_name: str, params: dict) -> tuple[str, frozenset]:
        """캐시 키를 생성한다.

        Args:
            indicator_name: 지표 이름 (예: 'RSI', 'MACD').
            params: 지표 파라미터 딕셔너리.

        Returns:
            (지표명, frozenset(params.items())) 튜플.
        """
        return (indicator_name, frozenset(params.items()))

    def get_or_compute(
        self,
        indicator_name: str,
        params: dict,
        df: pd.DataFrame,
        compute_fn: Callable[[pd.DataFrame, dict], IndicatorResult],
    ) -> IndicatorResult:
        """캐시에 결과가 있으면 반환, 없으면 계산 후 캐싱.

        Args:
            indicator_name: 지표 이름 (예: 'RSI', 'Supertrend').
            params: 지표 파라미터 딕셔너리.
            df: OHLCV DataFrame.
            compute_fn: 캐시 미스 시 호출할 계산 함수 (df, params) -> IndicatorResult.

        Returns:
            IndicatorResult: 캐시된 또는 새로 계산된 지표 결과.
        """
        key = self._make_key(indicator_name, params)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1

        result = compute_fn(df, params)

        with self._lock:
            self._cache[key] = result
        return result

    @property
    def hits(self) -> int:
        """캐시 히트 횟수."""
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        """캐시 미스 횟수."""
        with self._lock:
            return self._misses

    @property
    def size(self) -> int:
        """현재 캐시에 저장된 항목 수."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """캐시를 초기화하고 통계를 리셋한다."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
