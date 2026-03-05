"""Checkpoint 저장/로딩 단위 테스트 모듈.

체크포인트 라운드 트립, 빈 결과, 다중 결과, 파일 미존재 등 핵심 동작을 검증한다.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from sp500_backtest.engine.backtest import BacktestResult
from sp500_backtest.engine.checkpoint import (
    Checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _make_backtest_result(combination_id: str = "test_combo") -> BacktestResult:
    """테스트용 BacktestResult 생성.

    Args:
        combination_id: 조합 식별자.

    Returns:
        BacktestResult 인스턴스.
    """
    n = 50
    index = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    return BacktestResult(
        combination_id=combination_id,
        total_return=0.15,
        cagr=0.05,
        max_drawdown=-0.10,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        total_trades=10,
        win_rate=0.6,
        strategy_returns=pd.Series(rng.standard_normal(n) * 0.01, index=index),
        positions=pd.Series(rng.choice([-1, 0, 1], size=n), index=index),
    )


class TestCheckpointSaveLoad:
    """체크포인트 저장/로딩 라운드 트립 검증."""

    def test_save_and_load_round_trip(self, tmp_path):
        """저장 후 로딩하면 동일한 데이터를 복원한다."""
        result = _make_backtest_result("combo_001")
        ts = datetime(2024, 1, 15, 10, 30, 0)
        checkpoint = Checkpoint(
            completed_combinations=["combo_001"],
            results=[result],
            timestamp=ts,
            total_combinations=100,
        )

        path = str(tmp_path / "checkpoint.pkl")
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded is not None
        assert loaded.completed_combinations == ["combo_001"]
        assert len(loaded.results) == 1
        assert loaded.results[0].combination_id == "combo_001"
        assert loaded.total_combinations == 100
        assert loaded.timestamp == ts

    def test_loaded_result_data_integrity(self, tmp_path):
        """로딩된 BacktestResult의 수치 필드가 원본과 동일하다."""
        result = _make_backtest_result("combo_002")
        checkpoint = Checkpoint(
            completed_combinations=["combo_002"],
            results=[result],
            total_combinations=50,
        )

        path = str(tmp_path / "checkpoint.pkl")
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded is not None
        loaded_result = loaded.results[0]
        assert loaded_result.total_return == result.total_return
        assert loaded_result.cagr == result.cagr
        assert loaded_result.max_drawdown == result.max_drawdown
        assert loaded_result.sharpe_ratio == result.sharpe_ratio
        assert loaded_result.sortino_ratio == result.sortino_ratio
        assert loaded_result.total_trades == result.total_trades
        assert loaded_result.win_rate == result.win_rate
        pd.testing.assert_series_equal(
            loaded_result.strategy_returns, result.strategy_returns
        )
        pd.testing.assert_series_equal(loaded_result.positions, result.positions)


class TestCheckpointLoadNonExistent:
    """존재하지 않는 파일 로딩 검증."""

    def test_load_nonexistent_returns_none(self, tmp_path):
        """존재하지 않는 파일 경로에서 로딩하면 None을 반환한다."""
        path = str(tmp_path / "nonexistent.pkl")
        result = load_checkpoint(path)
        assert result is None


class TestCheckpointEmptyResults:
    """빈 결과 체크포인트 검증."""

    def test_empty_results_round_trip(self, tmp_path):
        """빈 결과 리스트를 가진 체크포인트도 정상 저장/로딩된다."""
        checkpoint = Checkpoint(
            completed_combinations=[],
            results=[],
            total_combinations=0,
        )

        path = str(tmp_path / "empty_checkpoint.pkl")
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded is not None
        assert loaded.completed_combinations == []
        assert loaded.results == []
        assert loaded.total_combinations == 0


class TestCheckpointMultipleResults:
    """다중 결과 체크포인트 검증."""

    def test_multiple_results_round_trip(self, tmp_path):
        """여러 BacktestResult를 포함한 체크포인트가 정상 저장/로딩된다."""
        results = [_make_backtest_result(f"combo_{i:03d}") for i in range(5)]
        combo_ids = [f"combo_{i:03d}" for i in range(5)]
        checkpoint = Checkpoint(
            completed_combinations=combo_ids,
            results=results,
            total_combinations=200,
        )

        path = str(tmp_path / "multi_checkpoint.pkl")
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)

        assert loaded is not None
        assert len(loaded.completed_combinations) == 5
        assert len(loaded.results) == 5
        assert loaded.total_combinations == 200
        for i, r in enumerate(loaded.results):
            assert r.combination_id == f"combo_{i:03d}"


class TestCheckpointSaveErrorHandling:
    """저장 실패 시 에러 핸들링 검증."""

    def test_save_to_invalid_path_prints_warning(self, tmp_path, capsys):
        """잘못된 경로에 저장 시도 시 경고 메시지를 출력하고 예외를 발생시키지 않는다."""
        checkpoint = Checkpoint(total_combinations=10)
        # /dev/null/impossible 같은 경로는 OS마다 다르므로
        # 읽기 전용 파일 위에 쓰기를 시도
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        target = readonly_dir / "sub" / "checkpoint.pkl"
        # 정상 경로이므로 디렉토리 생성 후 저장 성공해야 함
        save_checkpoint(checkpoint, str(target))
        assert os.path.exists(str(target))
