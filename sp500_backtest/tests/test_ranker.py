"""ResultRanker 단위 테스트.

rank(), save_csv(), print_summary() 메서드의 정렬 순서,
필수 컬럼, CSV 라운드 트립, 엣지 케이스를 검증한다.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.engine.backtest import BacktestResult
from sp500_backtest.results.ranker import REQUIRED_COLUMNS, ResultRanker


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------


def _make_index(n: int) -> pd.DatetimeIndex:
    """테스트용 DatetimeIndex 생성."""
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _make_result(
    combination_id: str = "combo_1",
    total_return: float = 0.10,
    cagr: float = 0.08,
    max_drawdown: float = -0.05,
    sharpe_ratio: float = 1.5,
    sortino_ratio: float = 2.0,
    total_trades: int = 10,
    win_rate: float = 0.6,
) -> BacktestResult:
    """테스트용 BacktestResult를 생성한다."""
    idx = _make_index(5)
    return BacktestResult(
        combination_id=combination_id,
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        total_trades=total_trades,
        win_rate=win_rate,
        strategy_returns=pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=idx),
        positions=pd.Series([0, 1, 1, -1, 0], index=idx, dtype=np.int64),
    )


def _make_results_list() -> list[BacktestResult]:
    """정렬 테스트용 BacktestResult 리스트 (3개)."""
    return [
        _make_result("A", total_return=0.05, cagr=0.04, sharpe_ratio=0.8),
        _make_result("B", total_return=0.20, cagr=0.15, sharpe_ratio=2.5),
        _make_result("C", total_return=0.10, cagr=0.08, sharpe_ratio=1.2),
    ]


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def ranker() -> ResultRanker:
    """ResultRanker 인스턴스."""
    return ResultRanker()


@pytest.fixture
def sample_results() -> list[BacktestResult]:
    """테스트용 결과 리스트."""
    return _make_results_list()


# ---------------------------------------------------------------------------
# rank() 테스트 — 필수 컬럼 검증
# ---------------------------------------------------------------------------


class TestRankColumns:
    """rank() 반환 DataFrame의 필수 컬럼 검증."""

    def test_required_columns_present(
        self, ranker: ResultRanker, sample_results: list[BacktestResult]
    ):
        """결과 테이블에 필수 컬럼이 모두 포함되는지 검증."""
        df = ranker.rank(sample_results)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"필수 컬럼 누락: {col}"

    def test_row_count_matches_input(
        self, ranker: ResultRanker, sample_results: list[BacktestResult]
    ):
        """행 수가 입력 결과 수와 일치하는지 검증."""
        df = ranker.rank(sample_results)
        assert len(df) == len(sample_results)


# ---------------------------------------------------------------------------
# rank() 테스트 — 정렬 순서 검증
# ---------------------------------------------------------------------------


class TestRankSortOrder:
    """rank() 정렬 순서(내림차순) 검증."""

    def test_sort_by_total_return_descending(
        self, ranker: ResultRanker, sample_results: list[BacktestResult]
    ):
        """total_return 기준 내림차순 정렬 검증."""
        df = ranker.rank(sample_results, sort_by="total_return")
        values = df["total_return"].tolist()
        assert values == sorted(values, reverse=True)
        # B(0.20) > C(0.10) > A(0.05)
        assert df.iloc[0]["combination_id"] == "B"
        assert df.iloc[1]["combination_id"] == "C"
        assert df.iloc[2]["combination_id"] == "A"

    def test_sort_by_cagr_descending(
        self, ranker: ResultRanker, sample_results: list[BacktestResult]
    ):
        """cagr 기준 내림차순 정렬 검증."""
        df = ranker.rank(sample_results, sort_by="cagr")
        values = df["cagr"].tolist()
        assert values == sorted(values, reverse=True)

    def test_sort_by_sharpe_ratio_descending(
        self, ranker: ResultRanker, sample_results: list[BacktestResult]
    ):
        """sharpe_ratio 기준 내림차순 정렬 검증."""
        df = ranker.rank(sample_results, sort_by="sharpe_ratio")
        values = df["sharpe_ratio"].tolist()
        assert values == sorted(values, reverse=True)
        # B(2.5) > C(1.2) > A(0.8)
        assert df.iloc[0]["combination_id"] == "B"

    def test_invalid_sort_by_raises_error(self, ranker: ResultRanker):
        """허용되지 않는 sort_by 값이면 ValueError 발생."""
        results = [_make_result()]
        with pytest.raises(ValueError, match="sort_by"):
            ranker.rank(results, sort_by="invalid_column")


# ---------------------------------------------------------------------------
# rank() 테스트 — 빈 결과 리스트
# ---------------------------------------------------------------------------


class TestRankEmpty:
    """빈 결과 리스트 처리 검증."""

    def test_empty_results_returns_empty_dataframe(self, ranker: ResultRanker):
        """빈 리스트 입력 시 빈 DataFrame 반환."""
        df = ranker.rank([])
        assert len(df) == 0
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_single_result(self, ranker: ResultRanker):
        """단일 결과도 정상 처리."""
        df = ranker.rank([_make_result("only_one")])
        assert len(df) == 1
        assert df.iloc[0]["combination_id"] == "only_one"


# ---------------------------------------------------------------------------
# save_csv() 및 라운드 트립 테스트
# ---------------------------------------------------------------------------


class TestSaveCsv:
    """CSV 저장 및 로딩 라운드 트립 검증."""

    def test_csv_round_trip(
        self, ranker: ResultRanker, sample_results: list[BacktestResult], tmp_path
    ):
        """CSV 저장 후 로딩하면 동일한 데이터 복원."""
        df_original = ranker.rank(sample_results)
        csv_path = str(tmp_path / "results.csv")

        ranker.save_csv(df_original, csv_path)
        df_loaded = pd.read_csv(csv_path)

        # 컬럼 동일
        assert list(df_loaded.columns) == list(df_original.columns)
        # 행 수 동일
        assert len(df_loaded) == len(df_original)
        # 값 비교 (float 근사 비교)
        for col in REQUIRED_COLUMNS:
            if col == "combination_id":
                assert df_loaded[col].tolist() == df_original[col].tolist()
            elif col == "total_trades":
                assert df_loaded[col].tolist() == df_original[col].tolist()
            else:
                np.testing.assert_allclose(
                    df_loaded[col].values,
                    df_original[col].values,
                    rtol=1e-6,
                    err_msg=f"컬럼 {col} 값 불일치",
                )

    def test_creates_output_directory(
        self, ranker: ResultRanker, sample_results: list[BacktestResult], tmp_path
    ):
        """출력 디렉토리가 없으면 자동 생성."""
        df = ranker.rank(sample_results)
        nested_path = str(tmp_path / "sub" / "dir" / "results.csv")

        ranker.save_csv(df, nested_path)
        assert os.path.exists(nested_path)

    def test_save_empty_dataframe(self, ranker: ResultRanker, tmp_path):
        """빈 DataFrame도 CSV로 저장 가능."""
        df = ranker.rank([])
        csv_path = str(tmp_path / "empty.csv")

        ranker.save_csv(df, csv_path)
        df_loaded = pd.read_csv(csv_path)
        assert len(df_loaded) == 0


# ---------------------------------------------------------------------------
# print_summary() 테스트
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """상위 N개 조합 요약 콘솔 출력 검증."""

    def test_print_summary_output(
        self,
        ranker: ResultRanker,
        sample_results: list[BacktestResult],
        capsys,
    ):
        """print_summary가 콘솔에 출력하는지 검증."""
        df = ranker.rank(sample_results)
        ranker.print_summary(df, top_n=2)

        captured = capsys.readouterr()
        # 상위 2개 출력 확인
        assert "상위 2개 조합 요약" in captured.out
        # rank 컬럼 포함 확인
        assert "rank" in captured.out

    def test_print_summary_respects_top_n(
        self,
        ranker: ResultRanker,
        sample_results: list[BacktestResult],
        capsys,
    ):
        """top_n보다 결과가 많으면 top_n개만 출력."""
        df = ranker.rank(sample_results)
        ranker.print_summary(df, top_n=1)

        captured = capsys.readouterr()
        assert "상위 1개 조합 요약" in captured.out

    def test_print_summary_empty_results(self, ranker: ResultRanker, capsys):
        """빈 결과 시 '결과가 없습니다' 메시지 출력."""
        df = ranker.rank([])
        ranker.print_summary(df)

        captured = capsys.readouterr()
        assert "결과가 없습니다" in captured.out

    def test_print_summary_top_n_larger_than_results(
        self,
        ranker: ResultRanker,
        sample_results: list[BacktestResult],
        capsys,
    ):
        """top_n이 결과 수보다 크면 전체 결과 출력."""
        df = ranker.rank(sample_results)
        ranker.print_summary(df, top_n=100)

        captured = capsys.readouterr()
        # 실제 결과 수(3)만큼 출력
        assert f"상위 {len(sample_results)}개 조합 요약" in captured.out
