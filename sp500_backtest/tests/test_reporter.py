"""ReportGenerator 단위 테스트.

파일명 생성 형식, 파일명 sanitization, 출력 디렉토리 생성,
오류 처리(QuantStats 예외 시 계속 진행), 콘솔 지표 출력을 검증한다.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.engine.backtest import BacktestResult
from sp500_backtest.results.reporter import ReportGenerator


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------


def _make_index(n: int = 5) -> pd.DatetimeIndex:
    """테스트용 DatetimeIndex 생성."""
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _make_result(
    combination_id: str = "RangeFilter_RQK_SuperTrend",
    total_return: float = 0.15,
    cagr: float = 0.10,
    max_drawdown: float = -0.08,
    sharpe_ratio: float = 1.8,
    sortino_ratio: float = 2.5,
    total_trades: int = 20,
    win_rate: float = 0.55,
) -> BacktestResult:
    """테스트용 BacktestResult를 생성한다."""
    idx = _make_index()
    return BacktestResult(
        combination_id=combination_id,
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        total_trades=total_trades,
        win_rate=win_rate,
        strategy_returns=pd.Series(
            [0.01, 0.02, -0.01, 0.03, 0.01], index=idx
        ),
        positions=pd.Series([0, 1, 1, -1, 0], index=idx, dtype=np.int64),
    )


def _make_benchmark(n: int = 5) -> pd.Series:
    """테스트용 벤치마크 수익률 Series 생성."""
    idx = _make_index(n)
    return pd.Series([0.005, 0.003, -0.002, 0.004, 0.001], index=idx)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def generator() -> ReportGenerator:
    """ReportGenerator 인스턴스."""
    return ReportGenerator()


@pytest.fixture
def sample_results() -> list[BacktestResult]:
    """테스트용 결과 리스트 (3개, 서로 다른 total_return)."""
    return [
        _make_result("Alpha_Strategy", total_return=0.05),
        _make_result("Beta_Strategy", total_return=0.20),
        _make_result("Gamma_Strategy", total_return=0.10),
    ]


@pytest.fixture
def benchmark() -> pd.Series:
    """테스트용 벤치마크 수익률."""
    return _make_benchmark()


# ---------------------------------------------------------------------------
# _generate_filename() 테스트
# ---------------------------------------------------------------------------


class TestGenerateFilename:
    """파일명 생성 형식 검증."""

    def test_basic_format(self, generator: ReportGenerator):
        """rank{:02d}_{id}.html 형식 검증."""
        filename = generator._generate_filename(1, "RangeFilter_RQK")
        assert filename == "rank01_RangeFilter_RQK.html"

    def test_rank_zero_padded(self, generator: ReportGenerator):
        """순위가 2자리 0-패딩되는지 검증."""
        assert generator._generate_filename(3, "Test").startswith("rank03_")
        assert generator._generate_filename(12, "Test").startswith("rank12_")

    def test_html_extension(self, generator: ReportGenerator):
        """파일명이 .html로 끝나는지 검증."""
        filename = generator._generate_filename(1, "combo")
        assert filename.endswith(".html")

    def test_special_chars_sanitized(self, generator: ReportGenerator):
        """특수문자가 언더스코어로 치환되는지 검증."""
        filename = generator._generate_filename(1, "A/B C+D")
        assert "/" not in filename
        assert " " not in filename
        assert "+" not in filename


# ---------------------------------------------------------------------------
# _sanitize_filename() 테스트
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    """파일명 sanitization 검증."""

    def test_spaces_replaced(self, generator: ReportGenerator):
        """공백이 언더스코어로 치환."""
        assert generator._sanitize_filename("hello world") == "hello_world"

    def test_slashes_replaced(self, generator: ReportGenerator):
        """슬래시가 언더스코어로 치환."""
        assert generator._sanitize_filename("a/b/c") == "a_b_c"

    def test_special_chars_replaced(self, generator: ReportGenerator):
        """특수문자가 언더스코어로 치환."""
        result = generator._sanitize_filename("a@b#c$d")
        assert result == "a_b_c_d"

    def test_consecutive_underscores_collapsed(self, generator: ReportGenerator):
        """연속 언더스코어가 하나로 축소."""
        result = generator._sanitize_filename("a___b")
        assert result == "a_b"

    def test_leading_trailing_underscores_stripped(
        self, generator: ReportGenerator
    ):
        """앞뒤 언더스코어 제거."""
        result = generator._sanitize_filename("__test__")
        assert result == "test"

    def test_truncation_at_100_chars(self, generator: ReportGenerator):
        """100자 초과 시 잘라냄."""
        long_id = "A" * 150
        result = generator._sanitize_filename(long_id)
        assert len(result) <= 100

    def test_alphanumeric_preserved(self, generator: ReportGenerator):
        """영문, 숫자, 언더스코어, 하이픈은 유지."""
        result = generator._sanitize_filename("abc-123_XYZ")
        assert result == "abc-123_XYZ"

    def test_empty_string(self, generator: ReportGenerator):
        """빈 문자열 입력 시 빈 문자열 반환."""
        result = generator._sanitize_filename("")
        assert result == ""


# ---------------------------------------------------------------------------
# generate() — 출력 디렉토리 생성 테스트
# ---------------------------------------------------------------------------


class TestGenerateOutputDir:
    """generate()가 출력 디렉토리를 자동 생성하는지 검증."""

    @patch("sp500_backtest.results.reporter.quantstats")
    def test_creates_output_directory(
        self,
        mock_qs: MagicMock,
        generator: ReportGenerator,
        benchmark: pd.Series,
        tmp_path,
    ):
        """존재하지 않는 출력 디렉토리를 자동 생성."""
        output_dir = str(tmp_path / "nested" / "reports")
        results = [_make_result()]

        mock_qs.reports.html = MagicMock()
        generator.generate(results, benchmark, top_n=1, output_dir=output_dir)

        assert os.path.isdir(output_dir)


# ---------------------------------------------------------------------------
# generate() — 오류 처리 테스트
# ---------------------------------------------------------------------------


class TestGenerateErrorHandling:
    """QuantStats 오류 시 로그 기록 후 다음 조합 계속 생성 검증."""

    @patch("sp500_backtest.results.reporter.quantstats")
    def test_continues_on_quantstats_error(
        self,
        mock_qs: MagicMock,
        generator: ReportGenerator,
        sample_results: list[BacktestResult],
        benchmark: pd.Series,
        tmp_path,
        capsys,
    ):
        """QuantStats 예외 발생 시 다음 조합으로 계속 진행."""
        output_dir = str(tmp_path / "reports")

        # 첫 번째 호출에서 예외, 나머지는 정상
        mock_qs.reports.html = MagicMock(
            side_effect=[RuntimeError("QuantStats error"), None, None]
        )

        generator.generate(
            sample_results, benchmark, top_n=3, output_dir=output_dir
        )

        # 3번 모두 호출 시도 (첫 번째 실패해도 나머지 계속)
        assert mock_qs.reports.html.call_count == 3

    @patch("sp500_backtest.results.reporter.quantstats")
    def test_logs_warning_on_error(
        self,
        mock_qs: MagicMock,
        generator: ReportGenerator,
        benchmark: pd.Series,
        tmp_path,
        caplog,
    ):
        """QuantStats 예외 시 경고 로그 기록."""
        import logging

        output_dir = str(tmp_path / "reports")
        results = [_make_result()]
        mock_qs.reports.html = MagicMock(
            side_effect=RuntimeError("test error")
        )

        with caplog.at_level(logging.WARNING):
            generator.generate(
                results, benchmark, top_n=1, output_dir=output_dir
            )

        assert "리포트 생성 실패" in caplog.text


# ---------------------------------------------------------------------------
# generate() — 콘솔 지표 출력 테스트
# ---------------------------------------------------------------------------


class TestGenerateConsoleOutput:
    """핵심 지표 콘솔 출력 검증."""

    @patch("sp500_backtest.results.reporter.quantstats")
    def test_prints_key_metrics(
        self,
        mock_qs: MagicMock,
        generator: ReportGenerator,
        benchmark: pd.Series,
        tmp_path,
        capsys,
    ):
        """총 수익률, CAGR, Max Drawdown, Sharpe, Sortino 콘솔 출력."""
        output_dir = str(tmp_path / "reports")
        results = [_make_result(total_return=0.15, cagr=0.10)]
        mock_qs.reports.html = MagicMock()

        generator.generate(results, benchmark, top_n=1, output_dir=output_dir)

        captured = capsys.readouterr()
        assert "총 수익률" in captured.out
        assert "CAGR" in captured.out
        assert "Max Drawdown" in captured.out
        assert "Sharpe Ratio" in captured.out
        assert "Sortino Ratio" in captured.out

    @patch("sp500_backtest.results.reporter.quantstats")
    def test_prints_rank_and_combination_id(
        self,
        mock_qs: MagicMock,
        generator: ReportGenerator,
        benchmark: pd.Series,
        tmp_path,
        capsys,
    ):
        """순위와 조합 ID가 콘솔에 출력."""
        output_dir = str(tmp_path / "reports")
        results = [_make_result("MyCombo")]
        mock_qs.reports.html = MagicMock()

        generator.generate(results, benchmark, top_n=1, output_dir=output_dir)

        captured = capsys.readouterr()
        assert "Rank 01" in captured.out
        assert "MyCombo" in captured.out


# ---------------------------------------------------------------------------
# generate() — top_n 및 정렬 테스트
# ---------------------------------------------------------------------------


class TestGenerateTopN:
    """상위 N개 선택 및 정렬 검증."""

    @patch("sp500_backtest.results.reporter.quantstats")
    def test_generates_only_top_n(
        self,
        mock_qs: MagicMock,
        generator: ReportGenerator,
        sample_results: list[BacktestResult],
        benchmark: pd.Series,
        tmp_path,
    ):
        """top_n=2이면 상위 2개만 리포트 생성."""
        output_dir = str(tmp_path / "reports")
        mock_qs.reports.html = MagicMock()

        generator.generate(
            sample_results, benchmark, top_n=2, output_dir=output_dir
        )

        assert mock_qs.reports.html.call_count == 2

    @patch("sp500_backtest.results.reporter.quantstats")
    def test_sorts_by_total_return_descending(
        self,
        mock_qs: MagicMock,
        generator: ReportGenerator,
        sample_results: list[BacktestResult],
        benchmark: pd.Series,
        tmp_path,
        capsys,
    ):
        """total_return 기준 내림차순으로 리포트 생성."""
        output_dir = str(tmp_path / "reports")
        mock_qs.reports.html = MagicMock()

        generator.generate(
            sample_results, benchmark, top_n=3, output_dir=output_dir
        )

        captured = capsys.readouterr()
        # Beta(0.20) > Gamma(0.10) > Alpha(0.05) 순서
        beta_pos = captured.out.find("Beta_Strategy")
        gamma_pos = captured.out.find("Gamma_Strategy")
        alpha_pos = captured.out.find("Alpha_Strategy")
        assert beta_pos < gamma_pos < alpha_pos
