"""ParameterOptimizer 단위 테스트 모듈.

그리드/랜덤 파라미터 생성, optimize 메서드, BacktestResult 반환 검증을 수행한다.
"""

import math

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.engine.backtest import BacktestResult
from sp500_backtest.engine.combination import IndicatorCombination
from sp500_backtest.engine.optimizer import (
    ParamSearchSpace,
    ParameterOptimizer,
    _format_eta,
)


# ---------------------------------------------------------------------------
# 테스트용 OHLCV DataFrame 생성 헬퍼
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame을 생성한다.

    Args:
        n: 데이터 행 수.

    Returns:
        OHLCV 컬럼을 가진 DataFrame.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10)  # 최소 가격 보장
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 1)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# ParamSearchSpace 기본 테스트
# ---------------------------------------------------------------------------

class TestParamSearchSpace:
    """ParamSearchSpace 데이터 클래스 테스트."""

    def test_creation(self) -> None:
        """ParamSearchSpace 인스턴스 생성 검증."""
        space = ParamSearchSpace(name="length", min_val=5.0, max_val=20.0, step=5.0)
        assert space.name == "length"
        assert space.min_val == 5.0
        assert space.max_val == 20.0
        assert space.step == 5.0


# ---------------------------------------------------------------------------
# ParameterOptimizer 초기화 테스트
# ---------------------------------------------------------------------------

class TestParameterOptimizerInit:
    """ParameterOptimizer 초기화 테스트."""

    def test_default_init(self) -> None:
        """기본 초기화 검증."""
        opt = ParameterOptimizer()
        assert opt.method == "grid"
        assert opt.n_workers == -1

    def test_custom_init(self) -> None:
        """커스텀 초기화 검증."""
        opt = ParameterOptimizer(method="random", n_workers=4)
        assert opt.method == "random"
        assert opt.n_workers == 4

    def test_invalid_method(self) -> None:
        """잘못된 method 값에 대한 ValueError 검증."""
        with pytest.raises(ValueError, match="method"):
            ParameterOptimizer(method="bayesian")


# ---------------------------------------------------------------------------
# 그리드 파라미터 생성 테스트
# ---------------------------------------------------------------------------

class TestGridParamGeneration:
    """_generate_grid_params 메서드 테스트."""

    def test_single_param_count(self) -> None:
        """단일 파라미터의 그리드 값 개수 검증: floor((max-min)/step)+1."""
        opt = ParameterOptimizer(method="grid")
        spaces = {
            "2 EMA Cross": [
                ParamSearchSpace(name="fast", min_val=10.0, max_val=50.0, step=10.0),
            ]
        }
        params = opt._generate_grid_params(spaces)
        # floor((50-10)/10)+1 = 5 값: 10, 20, 30, 40, 50
        assert len(params) == 5

    def test_single_param_values_in_range(self) -> None:
        """그리드 값이 [min, max] 범위 내에 있는지 검증."""
        opt = ParameterOptimizer(method="grid")
        spaces = {
            "RSI": [
                ParamSearchSpace(name="length", min_val=7.0, max_val=21.0, step=2.0),
            ]
        }
        params = opt._generate_grid_params(spaces)
        # floor((21-7)/2)+1 = 8 값: 7, 9, 11, 13, 15, 17, 19, 21
        assert len(params) == 8
        for ps in params:
            val = ps["RSI"]["length"]
            assert 7.0 <= val <= 21.0

    def test_multiple_params_cartesian_product(self) -> None:
        """여러 파라미터의 카르테시안 곱 개수 검증."""
        opt = ParameterOptimizer(method="grid")
        spaces = {
            "2 EMA Cross": [
                ParamSearchSpace(name="fast", min_val=10.0, max_val=30.0, step=10.0),  # 3개
                ParamSearchSpace(name="slow", min_val=100.0, max_val=200.0, step=50.0),  # 3개
            ]
        }
        params = opt._generate_grid_params(spaces)
        assert len(params) == 3 * 3  # 9개

    def test_multiple_indicators(self) -> None:
        """여러 지표의 파라미터 조합 검증."""
        opt = ParameterOptimizer(method="grid")
        spaces = {
            "2 EMA Cross": [
                ParamSearchSpace(name="fast", min_val=10.0, max_val=20.0, step=10.0),  # 2개
            ],
            "RSI": [
                ParamSearchSpace(name="length", min_val=7.0, max_val=14.0, step=7.0),  # 2개
            ],
        }
        params = opt._generate_grid_params(spaces)
        assert len(params) == 2 * 2  # 4개

    def test_empty_search_spaces(self) -> None:
        """빈 탐색 공간에 대해 빈 파라미터 세트 1개 반환 검증."""
        opt = ParameterOptimizer(method="grid")
        params = opt._generate_grid_params({})
        assert len(params) == 1
        assert params[0] == {}

    def test_grid_values_formula(self) -> None:
        """그리드 값 개수가 floor((max-min)/step)+1 공식과 일치하는지 검증."""
        opt = ParameterOptimizer(method="grid")
        space = ParamSearchSpace(name="x", min_val=1.0, max_val=10.0, step=3.0)
        values = opt._generate_grid_values(space)
        expected_count = math.floor((10.0 - 1.0) / 3.0) + 1  # 4
        assert len(values) == expected_count
        # 값: 1.0, 4.0, 7.0, 10.0
        assert values == [1.0, 4.0, 7.0, 10.0]


# ---------------------------------------------------------------------------
# 랜덤 파라미터 생성 테스트
# ---------------------------------------------------------------------------

class TestRandomParamGeneration:
    """_generate_random_params 메서드 테스트."""

    def test_correct_count(self) -> None:
        """랜덤 파라미터 세트 개수 검증."""
        opt = ParameterOptimizer(method="random")
        spaces = {
            "2 EMA Cross": [
                ParamSearchSpace(name="fast", min_val=5.0, max_val=50.0, step=5.0),
            ]
        }
        params = opt._generate_random_params(spaces, n=20)
        assert len(params) == 20

    def test_values_in_range(self) -> None:
        """랜덤 값이 [min, max] 범위 내에 있는지 검증."""
        opt = ParameterOptimizer(method="random")
        spaces = {
            "RSI": [
                ParamSearchSpace(name="length", min_val=7.0, max_val=21.0, step=1.0),
            ]
        }
        params = opt._generate_random_params(spaces, n=100)
        for ps in params:
            val = ps["RSI"]["length"]
            assert 7.0 <= val <= 21.0

    def test_empty_search_spaces(self) -> None:
        """빈 탐색 공간에 대해 빈 파라미터 세트 1개 반환 검증."""
        opt = ParameterOptimizer(method="random")
        params = opt._generate_random_params({}, n=5)
        # 탐색할 파라미터가 없으면 빈 세트 1개만 반환
        assert len(params) == 1
        assert params[0] == {}



# ---------------------------------------------------------------------------
# optimize 메서드 통합 테스트
# ---------------------------------------------------------------------------

class TestOptimize:
    """optimize 메서드 통합 테스트 (실제 지표 사용)."""

    def test_grid_optimize_with_two_ema_cross(self) -> None:
        """2 EMA Cross 리딩 지표로 그리드 최적화 수행 및 결과 검증."""
        df = _make_ohlcv(200)
        combination = IndicatorCombination(
            id="TwoEMACross",
            leading="2 EMA Cross",
            leading_params={},
            confirmations=[],
        )
        search_spaces = {
            "2 EMA Cross": [
                ParamSearchSpace(name="fast", min_val=10.0, max_val=20.0, step=10.0),  # 2개
                ParamSearchSpace(name="slow", min_val=50.0, max_val=100.0, step=50.0),  # 2개
            ]
        }
        opt = ParameterOptimizer(method="grid", n_workers=1)
        results = opt.optimize(
            combination=combination,
            df=df,
            search_spaces=search_spaces,
            signal_expiry=3,
            alternate_signal=True,
            transaction_cost=0.001,
        )
        # 2×2 = 4개 파라미터 조합
        assert len(results) == 4
        for r in results:
            assert isinstance(r, BacktestResult)

    def test_random_optimize_with_two_ema_cross(self) -> None:
        """2 EMA Cross 리딩 지표로 랜덤 최적화 수행 및 결과 검증."""
        df = _make_ohlcv(200)
        combination = IndicatorCombination(
            id="TwoEMACross",
            leading="2 EMA Cross",
            leading_params={},
            confirmations=[],
        )
        search_spaces = {
            "2 EMA Cross": [
                ParamSearchSpace(name="fast", min_val=5.0, max_val=50.0, step=5.0),
            ]
        }
        opt = ParameterOptimizer(method="random", n_workers=1)
        results = opt.optimize(
            combination=combination,
            df=df,
            search_spaces=search_spaces,
            random_iterations=5,
        )
        assert len(results) == 5
        for r in results:
            assert isinstance(r, BacktestResult)

    def test_optimize_empty_search_spaces(self) -> None:
        """빈 탐색 공간으로 최적화 시 기본 파라미터 1개 결과 반환 검증."""
        df = _make_ohlcv(200)
        combination = IndicatorCombination(
            id="TwoEMACross",
            leading="2 EMA Cross",
            leading_params={},
            confirmations=[],
        )
        opt = ParameterOptimizer(method="grid", n_workers=1)
        results = opt.optimize(
            combination=combination,
            df=df,
            search_spaces={},
        )
        # 빈 탐색 공간 → 기본 파라미터 1개 조합
        assert len(results) == 1
        assert isinstance(results[0], BacktestResult)

    def test_optimize_results_have_combination_id(self) -> None:
        """결과의 combination_id에 파라미터 정보가 포함되는지 검증."""
        df = _make_ohlcv(200)
        combination = IndicatorCombination(
            id="TwoEMACross",
            leading="2 EMA Cross",
            leading_params={},
            confirmations=[],
        )
        search_spaces = {
            "2 EMA Cross": [
                ParamSearchSpace(name="fast", min_val=10.0, max_val=10.0, step=10.0),  # 1개
            ]
        }
        opt = ParameterOptimizer(method="grid", n_workers=1)
        results = opt.optimize(combination=combination, df=df, search_spaces=search_spaces)
        assert len(results) == 1
        assert "TwoEMACross" in results[0].combination_id


# ---------------------------------------------------------------------------
# 유틸리티 함수 테스트
# ---------------------------------------------------------------------------

class TestFormatEta:
    """_format_eta 유틸리티 함수 테스트."""

    def test_zero_seconds(self) -> None:
        assert _format_eta(0) == "00:00:00"

    def test_90_seconds(self) -> None:
        assert _format_eta(90) == "00:01:30"

    def test_3661_seconds(self) -> None:
        assert _format_eta(3661) == "01:01:01"

    def test_negative(self) -> None:
        assert _format_eta(-1) == "??:??:??"

    def test_inf(self) -> None:
        assert _format_eta(float("inf")) == "??:??:??"
