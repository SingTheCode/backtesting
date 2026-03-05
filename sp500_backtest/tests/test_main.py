"""메인 오케스트레이터 단위 테스트.

run_pipeline 함수, 지표 레지스트리 구축, 조합 생성,
전체 파이프라인 mock 기반 통합 테스트를 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.config import load_config
from sp500_backtest.engine.backtest import BacktestResult
from sp500_backtest.engine.combination import IndicatorCombination
from sp500_backtest.main import (
    _build_search_spaces,
    _run_combination_backtest,
    build_confirmation_info,
    build_leading_names,
    run_pipeline,
)
from sp500_backtest.engine.optimizer import ParamSearchSpace


# ── 헬퍼 ──


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame을 생성한다.

    Args:
        n: 행 수 (거래일 수).

    Returns:
        OHLCV DataFrame (Open, High, Low, Close, Volume).
    """
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10.0)  # 양수 보장
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + abs(rng.normal(0, 1, n)),
            "Low": close - abs(rng.normal(0, 1, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=pd.date_range("2021-01-01", periods=n, freq="B"),
    )


def _make_backtest_result(combo_id: str = "test_combo") -> BacktestResult:
    """테스트용 BacktestResult를 생성한다.

    Args:
        combo_id: 조합 식별자.

    Returns:
        BacktestResult 인스턴스.
    """
    n = 50
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return BacktestResult(
        combination_id=combo_id,
        total_return=0.15,
        cagr=0.05,
        max_drawdown=-0.10,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        total_trades=10,
        win_rate=0.6,
        strategy_returns=pd.Series(np.random.default_rng(0).normal(0, 0.01, n), index=idx),
        positions=pd.Series(np.random.default_rng(0).choice([1, -1, 0], n), index=idx),
    )


# ── run_pipeline 존재 및 호출 가능 테스트 ──


class TestPipelineFunctionExists:
    """run_pipeline 함수 존재 및 호출 가능 여부 검증."""

    def test_run_pipeline_is_callable(self) -> None:
        """run_pipeline이 호출 가능한 함수인지 확인한다."""
        assert callable(run_pipeline)

    def test_main_module_has_main_function(self) -> None:
        """main 모듈에 main() 엔트리포인트가 존재하는지 확인한다."""
        from sp500_backtest.main import main

        assert callable(main)


# ── 설정 로딩 통합 테스트 ──


class TestConfigIntegration:
    """config.yaml 로딩 통합 검증."""

    def test_load_config_returns_dict(self) -> None:
        """load_config가 딕셔너리를 반환하는지 확인한다."""
        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_required_keys(self) -> None:
        """설정에 필수 키가 포함되어 있는지 확인한다."""
        config = load_config()
        required_keys = ["data", "combination", "backtest", "optimizer", "results", "performance"]
        for key in required_keys:
            assert key in config, f"설정에 '{key}' 키가 없습니다"


# ── 지표 레지스트리 구축 테스트 ──


class TestIndicatorRegistry:
    """지표 레지스트리 구축 검증."""

    def test_leading_names_count(self) -> None:
        """리딩 지표가 37개인지 확인한다."""
        names = build_leading_names()
        assert len(names) == 37, f"리딩 지표 수: {len(names)} (기대: 37)"

    def test_leading_names_are_strings(self) -> None:
        """리딩 지표 이름이 모두 문자열인지 확인한다."""
        names = build_leading_names()
        for name in names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_confirmation_info_count_positive(self) -> None:
        """확인 지표가 1개 이상인지 확인한다."""
        info = build_confirmation_info()
        assert len(info) > 0, "확인 지표가 0개입니다"

    def test_confirmation_info_structure(self) -> None:
        """확인 지표 정보가 올바른 구조인지 확인한다."""
        info = build_confirmation_info()
        for item in info:
            assert "name" in item, "확인 지표에 'name' 키가 없습니다"
            assert "subtypes" in item, "확인 지표에 'subtypes' 키가 없습니다"
            assert isinstance(item["name"], str)
            assert isinstance(item["subtypes"], list)


# ── 파라미터 탐색 공간 변환 테스트 ──


class TestBuildSearchSpaces:
    """_build_search_spaces 함수 검증."""

    def test_empty_param_ranges(self) -> None:
        """빈 param_ranges가 빈 딕셔너리를 반환하는지 확인한다."""
        result = _build_search_spaces({})
        assert result == {}

    def test_single_indicator_conversion(self) -> None:
        """단일 지표 param_ranges가 올바르게 변환되는지 확인한다."""
        param_ranges = {
            "rsi": {
                "length": {"min": 7, "max": 21, "step": 2},
            }
        }
        result = _build_search_spaces(param_ranges)
        assert "rsi" in result
        assert len(result["rsi"]) == 1
        space = result["rsi"][0]
        assert isinstance(space, ParamSearchSpace)
        assert space.name == "length"
        assert space.min_val == 7.0
        assert space.max_val == 21.0
        assert space.step == 2.0


# ── 조합 생성 소규모 테스트 ──


class TestCombinationGeneration:
    """소규모 데이터로 조합 생성 검증."""

    def test_small_combination_generation(self) -> None:
        """리딩 2개, 확인 1개로 조합이 올바르게 생성되는지 확인한다."""
        from sp500_backtest.engine.combination import CombinationEngine

        leading_names = ["Indicator A", "Indicator B"]
        confirmation_info = [
            {"name": "Conf X", "subtypes": []},
        ]
        engine = CombinationEngine(
            leading_names=leading_names,
            confirmation_info=confirmation_info,
        )
        combos = engine.generate(max_confirmations=1, max_combinations=100)
        # 2 리딩 × (C(1,0) + C(1,1)) = 2 × 2 = 4
        assert len(combos) == 4

    def test_combination_ids_unique(self) -> None:
        """생성된 조합 ID가 모두 고유한지 확인한다."""
        from sp500_backtest.engine.combination import CombinationEngine

        leading_names = ["A", "B", "C"]
        confirmation_info = [
            {"name": "X Confirmation", "subtypes": ["S1", "S2"]},
            {"name": "Y Confirmation", "subtypes": []},
        ]
        engine = CombinationEngine(
            leading_names=leading_names,
            confirmation_info=confirmation_info,
        )
        combos = engine.generate(max_confirmations=2, max_combinations=10000)
        ids = [c.id for c in combos]
        assert len(ids) == len(set(ids)), "조합 ID에 중복이 있습니다"


# ── 단일 조합 백테스팅 테스트 ──


class TestRunCombinationBacktest:
    """_run_combination_backtest 함수 검증."""

    @patch("sp500_backtest.main._run_single_backtest")
    def test_no_param_ranges_calls_single_backtest(self, mock_single: MagicMock) -> None:
        """param_ranges가 없으면 _run_single_backtest를 호출하는지 확인한다."""
        mock_single.return_value = _make_backtest_result()
        combo = IndicatorCombination(
            id="test",
            leading="Range Filter",
            leading_params={},
            confirmations=[],
        )
        config = {
            "backtest": {"signal_expiry": 3, "alternate_signal": True, "transaction_cost": 0.001},
            "optimizer": {"method": "grid"},
            "performance": {"n_workers": 1},
        }
        from sp500_backtest.engine.cache import IndicatorCache

        results = _run_combination_backtest(
            combination=combo,
            df=_make_ohlcv(50),
            config=config,
            search_spaces={},
            cache=IndicatorCache(),
        )
        assert len(results) == 1
        mock_single.assert_called_once()


# ── 전체 파이프라인 mock 기반 통합 테스트 ──


class TestFullPipelineMocked:
    """DataFetcher를 mock하여 전체 파이프라인 흐름을 검증한다."""

    @patch("sp500_backtest.main.ReportGenerator")
    @patch("sp500_backtest.main.DataFetcher")
    def test_pipeline_runs_with_minimal_config(
        self, mock_fetcher_cls: MagicMock, mock_reporter_cls: MagicMock
    ) -> None:
        """최소 설정으로 파이프라인이 정상 실행되는지 확인한다.

        DataFetcher와 ReportGenerator를 mock하여 네트워크 호출 없이 테스트한다.
        리딩 지표 1개, 확인 지표 0개, max_confirmations=0으로 최소 조합만 생성한다.
        """
        # DataFetcher mock 설정
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = _make_ohlcv(200)
        mock_fetcher_cls.return_value = mock_fetcher

        # ReportGenerator mock 설정 (QuantStats 의존성 회피)
        mock_reporter = MagicMock()
        mock_reporter_cls.return_value = mock_reporter

        # 최소 설정: 조합 수를 극도로 제한
        leading_names = build_leading_names()
        config = {
            "data": {"symbol": "^GSPC", "period": "3y"},
            "combination": {
                "max_confirmations": 0,  # 확인 지표 없이 리딩만
                "max_combinations": 50,  # 최대 50개 조합
            },
            "backtest": {
                "transaction_cost": 0.001,
                "signal_expiry": 3,
                "alternate_signal": True,
            },
            "optimizer": {"method": "grid", "random_iterations": 10},
            "results": {
                "sort_by": "total_return",
                "top_n_display": 5,
                "top_n_report": 2,
            },
            "performance": {
                "n_workers": 1,
                "checkpoint_interval": 1000,  # 체크포인트 비활성화 (큰 간격)
            },
            "param_ranges": {},  # 파라미터 최적화 없음
        }

        results = run_pipeline(config=config)

        # 검증: DataFetcher.fetch가 호출됨
        mock_fetcher.fetch.assert_called_once()

        # 검증: 결과가 리스트로 반환됨
        assert isinstance(results, list)

        # 검증: 결과가 존재함 (37개 리딩 × 1 = 37개, 단 max_combinations=50이므로 ≤50)
        assert len(results) > 0

        # 검증: 각 결과가 BacktestResult 인스턴스
        for r in results:
            assert isinstance(r, BacktestResult)

    @patch("sp500_backtest.main.ReportGenerator")
    @patch("sp500_backtest.main.DataFetcher")
    def test_pipeline_handles_empty_results_gracefully(
        self, mock_fetcher_cls: MagicMock, mock_reporter_cls: MagicMock
    ) -> None:
        """데이터가 매우 짧아 결과가 적을 때도 파이프라인이 정상 종료되는지 확인한다."""
        # 매우 짧은 데이터 (10일)
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = _make_ohlcv(10)
        mock_fetcher_cls.return_value = mock_fetcher

        mock_reporter = MagicMock()
        mock_reporter_cls.return_value = mock_reporter

        config = {
            "data": {"symbol": "^GSPC", "period": "1y"},
            "combination": {"max_confirmations": 0, "max_combinations": 5},
            "backtest": {"transaction_cost": 0.0, "signal_expiry": 3, "alternate_signal": True},
            "optimizer": {"method": "grid", "random_iterations": 5},
            "results": {"sort_by": "total_return", "top_n_display": 3, "top_n_report": 1},
            "performance": {"n_workers": 1, "checkpoint_interval": 10000},
            "param_ranges": {},
        }

        # 예외 없이 실행 완료되어야 함
        results = run_pipeline(config=config)
        assert isinstance(results, list)
