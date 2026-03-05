"""파라미터 최적화 엔진 모듈.

그리드 서치(Grid Search) 및 랜덤 서치(Random Search) 방식으로
지표 조합의 파라미터를 탐색하고, 병렬 백테스팅을 수행한다.
진행률 및 ETA를 콘솔에 출력한다.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sp500_backtest.engine.backtest import BacktestEngine, BacktestResult
from sp500_backtest.engine.combination import IndicatorCombination
from sp500_backtest.engine.signal import SignalGenerator
from sp500_backtest.indicators.base import ConfirmationIndicator, IndicatorResult

if TYPE_CHECKING:
    from sp500_backtest.indicators.base import BaseIndicator


@dataclass
class ParamSearchSpace:
    """파라미터 탐색 공간 정의 데이터 클래스."""

    name: str  # 파라미터 이름 (예: 'fast', 'slow', 'length')
    min_val: float  # 최소값
    max_val: float  # 최대값
    step: float  # 스텝 크기


def _build_leading_registry() -> dict[str, type[BaseIndicator]]:
    """리딩 지표 이름 → 클래스 매핑 레지스트리를 생성한다.

    Returns:
        리딩 지표 이름을 키로, 클래스를 값으로 하는 딕셔너리.
    """
    from sp500_backtest.indicators import leading as _mod

    from sp500_backtest.indicators.base import BaseIndicator as _Base

    registry: dict[str, type[BaseIndicator]] = {}
    for attr_name in dir(_mod):
        obj = getattr(_mod, attr_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, _Base)
            and obj is not _Base
            and not issubclass(obj, ConfirmationIndicator)
        ):
            instance = obj()
            registry[instance.name] = obj
    return registry


def _build_confirmation_registry() -> dict[str, type[ConfirmationIndicator]]:
    """확인 지표 이름 → 클래스 매핑 레지스트리를 생성한다.

    Returns:
        확인 지표 이름을 키로, 클래스를 값으로 하는 딕셔너리.
    """
    from sp500_backtest.indicators import confirmation as _mod

    registry: dict[str, type[ConfirmationIndicator]] = {}
    for attr_name in dir(_mod):
        obj = getattr(_mod, attr_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, ConfirmationIndicator)
            and obj is not ConfirmationIndicator
        ):
            instance = obj()
            registry[instance.name] = obj
    return registry


# 모듈 레벨 레지스트리 (지연 초기화)
_leading_registry: dict[str, type[BaseIndicator]] | None = None
_confirmation_registry: dict[str, type[ConfirmationIndicator]] | None = None


def _get_leading_registry() -> dict[str, type[BaseIndicator]]:
    """리딩 지표 레지스트리를 반환한다 (지연 초기화)."""
    global _leading_registry
    if _leading_registry is None:
        _leading_registry = _build_leading_registry()
    return _leading_registry


def _get_confirmation_registry() -> dict[str, type[ConfirmationIndicator]]:
    """확인 지표 레지스트리를 반환한다 (지연 초기화)."""
    global _confirmation_registry
    if _confirmation_registry is None:
        _confirmation_registry = _build_confirmation_registry()
    return _confirmation_registry


def _run_single_backtest(
    combination: IndicatorCombination,
    df: pd.DataFrame,
    param_set: dict,
    signal_expiry: int,
    alternate_signal: bool,
    transaction_cost: float,
) -> BacktestResult | None:
    """단일 파라미터 세트에 대해 지표 계산 → 시그널 생성 → 백테스팅을 수행한다.

    Args:
        combination: 지표 조합 정의.
        df: OHLCV DataFrame.
        param_set: 파라미터 딕셔너리 (지표별 파라미터 포함).
        signal_expiry: 시그널 만료 캔들 수.
        alternate_signal: 연속 동일 방향 시그널 필터링 여부.
        transaction_cost: 거래 비용 비율.

    Returns:
        BacktestResult 또는 오류 시 None.
    """
    try:
        leading_reg = _get_leading_registry()
        confirmation_reg = _get_confirmation_registry()

        # 리딩 지표 계산
        leading_cls = leading_reg.get(combination.leading)
        if leading_cls is None:
            return None
        leading_indicator = leading_cls()

        # 리딩 지표 파라미터: combination 기본 파라미터 + param_set에서 리딩 지표용 파라미터 추출
        leading_params = dict(combination.leading_params)
        leading_search_key = combination.leading  # 리딩 지표 이름으로 검색
        if leading_search_key in param_set:
            leading_params.update(param_set[leading_search_key])

        leading_result: IndicatorResult = leading_indicator.calculate(df, leading_params)

        # 확인 지표 계산
        confirmation_results: list[IndicatorResult] = []
        for conf_info in combination.confirmations:
            conf_name = conf_info["name"]
            conf_subtype = conf_info.get("subtype")
            conf_params = dict(conf_info.get("params", {}))

            # param_set에서 확인 지표용 파라미터 추출
            if conf_name in param_set:
                conf_params.update(param_set[conf_name])

            conf_cls = confirmation_reg.get(conf_name)
            if conf_cls is None:
                return None
            conf_indicator = conf_cls()
            conf_result = conf_indicator.calculate(df, conf_params, subtype=conf_subtype)
            confirmation_results.append(conf_result)

        # 시그널 생성
        signal_gen = SignalGenerator()
        positions = signal_gen.generate(
            leading_result,
            confirmation_results,
            signal_expiry=signal_expiry,
            alternate_signal=alternate_signal,
        )

        # 백테스팅 수행
        engine = BacktestEngine()
        # 조합 ID에 파라미터 정보 추가
        param_suffix = _format_param_suffix(param_set)
        combo_id = f"{combination.id}|{param_suffix}" if param_suffix else combination.id

        result = engine.run(
            positions=positions,
            prices=df["Close"],
            transaction_cost=transaction_cost,
            combination_id=combo_id,
        )
        return result

    except Exception:
        return None


def _format_param_suffix(param_set: dict) -> str:
    """파라미터 세트를 간결한 문자열로 변환한다.

    Args:
        param_set: 지표별 파라미터 딕셔너리.

    Returns:
        파라미터 요약 문자열 (예: "fast=10,slow=50").
    """
    parts: list[str] = []
    for indicator_name, params in param_set.items():
        if isinstance(params, dict):
            for k, v in params.items():
                # 정수로 표현 가능하면 정수로
                if isinstance(v, float) and v == int(v):
                    parts.append(f"{k}={int(v)}")
                else:
                    parts.append(f"{k}={v}")
    return ",".join(parts)


def _format_eta(seconds: float) -> str:
    """초 단위 시간을 HH:MM:SS 형식으로 변환한다.

    Args:
        seconds: 남은 시간 (초).

    Returns:
        "HH:MM:SS" 형식 문자열.
    """
    if seconds < 0 or not math.isfinite(seconds):
        return "??:??:??"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ParameterOptimizer:
    """파라미터 최적화 엔진.

    그리드 서치 또는 랜덤 서치 방식으로 지표 조합의 파라미터를 탐색하고,
    ThreadPoolExecutor를 사용하여 병렬 백테스팅을 수행한다.
    """

    def __init__(self, method: str = "grid", n_workers: int = -1) -> None:
        """ParameterOptimizer 초기화.

        Args:
            method: 탐색 방식 ("grid" 또는 "random").
            n_workers: 병렬 워커 수 (-1이면 CPU 코어 수 사용).

        Raises:
            ValueError: method가 "grid" 또는 "random"이 아닌 경우.
        """
        if method not in ("grid", "random"):
            raise ValueError(f"method는 'grid' 또는 'random'이어야 합니다: {method}")
        self._method = method  # 탐색 방식
        self._n_workers = n_workers  # 병렬 워커 수

    @property
    def method(self) -> str:
        """현재 탐색 방식."""
        return self._method

    @property
    def n_workers(self) -> int:
        """병렬 워커 수."""
        return self._n_workers

    def optimize(
        self,
        combination: IndicatorCombination,
        df: pd.DataFrame,
        search_spaces: dict[str, list[ParamSearchSpace]],
        signal_expiry: int = 3,
        alternate_signal: bool = True,
        transaction_cost: float = 0.001,
        random_iterations: int = 1000,
    ) -> list[BacktestResult]:
        """주어진 조합에 대해 파라미터 최적화를 수행한다.

        Args:
            combination: 지표 조합 정의.
            df: OHLCV DataFrame.
            search_spaces: 지표별 파라미터 탐색 공간.
                키: 지표 이름, 값: ParamSearchSpace 리스트.
            signal_expiry: 시그널 만료 캔들 수 (기본값: 3).
            alternate_signal: 연속 동일 방향 시그널 필터링 여부 (기본값: True).
            transaction_cost: 거래 비용 비율 (기본값: 0.001).
            random_iterations: 랜덤 서치 시 반복 횟수 (기본값: 1000).

        Returns:
            모든 파라미터 조합의 BacktestResult 리스트.
        """
        # 파라미터 조합 생성
        if self._method == "grid":
            param_sets = self._generate_grid_params(search_spaces)
        else:
            param_sets = self._generate_random_params(search_spaces, random_iterations)

        total = len(param_sets)
        if total == 0:
            return []

        print(f"파라미터 최적화 시작: {self._method} 방식, 총 {total}개 조합")

        results: list[BacktestResult] = []
        completed = 0
        start_time = time.time()

        # 워커 수 결정
        import os
        workers = self._n_workers if self._n_workers > 0 else os.cpu_count() or 1

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_single_backtest,
                    combination,
                    df,
                    ps,
                    signal_expiry,
                    alternate_signal,
                    transaction_cost,
                ): ps
                for ps in param_sets
            }

            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result is not None:
                    results.append(result)

                # 진행률 및 ETA 출력 (10% 간격 또는 마지막)
                if completed % max(1, total // 10) == 0 or completed == total:
                    elapsed = time.time() - start_time
                    if completed > 0:
                        eta_seconds = (elapsed / completed) * (total - completed)
                    else:
                        eta_seconds = 0.0
                    pct = completed / total * 100
                    print(
                        f"진행: {completed}/{total} ({pct:.1f}%) | "
                        f"ETA: {_format_eta(eta_seconds)}"
                    )

        print(f"파라미터 최적화 완료: 유효 결과 {len(results)}/{total}개")
        return results

    def _generate_grid_params(
        self, search_spaces: dict[str, list[ParamSearchSpace]]
    ) -> list[dict]:
        """그리드 서치용 모든 파라미터 조합을 생성한다.

        각 ParamSearchSpace의 [min_val, max_val] 범위를 step 간격으로 나누어
        모든 조합을 생성한다.

        Args:
            search_spaces: 지표별 파라미터 탐색 공간.

        Returns:
            파라미터 딕셔너리 리스트. 각 항목은 {지표이름: {파라미터명: 값}} 형태.
        """
        if not search_spaces:
            return [{}]

        # 각 파라미터의 가능한 값 목록 생성
        param_keys: list[tuple[str, str]] = []  # (지표이름, 파라미터이름)
        param_values: list[list[float]] = []  # 각 파라미터의 값 목록

        for indicator_name, spaces in search_spaces.items():
            for space in spaces:
                param_keys.append((indicator_name, space.name))
                values = self._generate_grid_values(space)
                param_values.append(values)

        if not param_keys:
            return [{}]

        # 모든 조합 생성 (cartesian product)
        result: list[dict] = []
        for combo in product(*param_values):
            param_set: dict[str, dict] = {}
            for (ind_name, param_name), val in zip(param_keys, combo):
                if ind_name not in param_set:
                    param_set[ind_name] = {}
                param_set[ind_name][param_name] = val
            result.append(param_set)

        return result

    @staticmethod
    def _generate_grid_values(space: ParamSearchSpace) -> list[float]:
        """단일 ParamSearchSpace에서 그리드 값 목록을 생성한다.

        값 개수: floor((max_val - min_val) / step) + 1

        Args:
            space: 파라미터 탐색 공간 정의.

        Returns:
            [min_val, min_val+step, ..., max_val 이하] 값 리스트.
        """
        count = int(math.floor((space.max_val - space.min_val) / space.step)) + 1
        values: list[float] = []
        for i in range(count):
            val = space.min_val + i * space.step
            # 부동소수점 오차 보정: max_val 초과 방지
            if val > space.max_val + 1e-9:
                break
            values.append(round(val, 10))
        return values

    def _generate_random_params(
        self, search_spaces: dict[str, list[ParamSearchSpace]], n: int
    ) -> list[dict]:
        """랜덤 서치용 파라미터 세트를 n개 샘플링한다.

        각 파라미터는 [min_val, max_val] 범위 내에서 균일 분포로 샘플링한다.

        Args:
            search_spaces: 지표별 파라미터 탐색 공간.
            n: 샘플링할 파라미터 세트 수.

        Returns:
            파라미터 딕셔너리 리스트.
        """
        if not search_spaces:
            return [{}]

        # 파라미터 키 목록 수집
        param_keys: list[tuple[str, str, ParamSearchSpace]] = []
        for indicator_name, spaces in search_spaces.items():
            for space in spaces:
                param_keys.append((indicator_name, space.name, space))

        if not param_keys:
            return [{}]

        result: list[dict] = []
        for _ in range(n):
            param_set: dict[str, dict] = {}
            for ind_name, param_name, space in param_keys:
                # step 단위로 스냅하여 범위 내 값 생성
                num_steps = int(math.floor((space.max_val - space.min_val) / space.step))
                step_idx = random.randint(0, max(0, num_steps))
                val = round(space.min_val + step_idx * space.step, 10)
                # max_val 초과 방지
                val = min(val, space.max_val)
                if ind_name not in param_set:
                    param_set[ind_name] = {}
                param_set[ind_name][param_name] = val
            result.append(param_set)

        return result
