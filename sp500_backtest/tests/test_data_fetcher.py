"""DataFetcher 단위 테스트.

yfinance 데이터 수집, 결측값 처리, 재시도 로직, 콘솔 출력을 검증한다.
실제 네트워크 호출 없이 mock을 사용하여 테스트한다.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.data.fetcher import DataFetcher


# ---------------------------------------------------------------------------
# 테스트 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def fetcher() -> DataFetcher:
    """DataFetcher 인스턴스를 생성한다."""
    return DataFetcher()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """정상적인 OHLCV DataFrame을 생성한다."""
    dates = pd.bdate_range("2022-01-03", periods=100)
    np.random.seed(42)
    close = 4500 + np.cumsum(np.random.randn(100) * 10)
    return pd.DataFrame(
        {
            "Open": close - np.random.rand(100) * 5,
            "High": close + np.random.rand(100) * 10,
            "Low": close - np.random.rand(100) * 10,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 5_000_000, size=100),
        },
        index=dates,
    )


@pytest.fixture
def ohlcv_with_nans(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """일부 위치에 NaN이 포함된 OHLCV DataFrame을 생성한다."""
    df = sample_ohlcv_df.copy()
    df.iloc[10, 0] = np.nan  # Open
    df.iloc[20, 3] = np.nan  # Close
    df.iloc[30, 4] = np.nan  # Volume
    df.iloc[50, 1] = np.nan  # High
    df.iloc[50, 2] = np.nan  # Low
    return df


# ---------------------------------------------------------------------------
# 정상 데이터 수집 테스트
# ---------------------------------------------------------------------------


class TestFetchSuccess:
    """정상적인 데이터 수집 시나리오 검증."""

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_fetch_returns_dataframe(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """fetch()가 pd.DataFrame을 반환한다."""
        mock_download.return_value = sample_ohlcv_df
        result = fetcher.fetch()
        assert isinstance(result, pd.DataFrame)

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_fetch_has_required_columns(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """반환된 DataFrame이 OHLCV 5개 컬럼을 포함한다."""
        mock_download.return_value = sample_ohlcv_df
        result = fetcher.fetch()
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_fetch_has_datetime_index(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """반환된 DataFrame의 인덱스가 DatetimeIndex이다."""
        mock_download.return_value = sample_ohlcv_df
        result = fetcher.fetch()
        assert isinstance(result.index, pd.DatetimeIndex)

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_fetch_preserves_row_count(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """반환된 DataFrame의 행 수가 원본과 동일하다."""
        mock_download.return_value = sample_ohlcv_df
        result = fetcher.fetch()
        assert len(result) == len(sample_ohlcv_df)

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_fetch_with_custom_symbol_and_period(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """사용자 지정 심볼과 기간으로 호출할 수 있다."""
        mock_download.return_value = sample_ohlcv_df
        result = fetcher.fetch(symbol="AAPL", period="1y")
        mock_download.assert_called_once_with(
            "AAPL", period="1y", auto_adjust=True, progress=False
        )
        assert isinstance(result, pd.DataFrame)

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_fetch_default_params(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """기본 파라미터로 ^GSPC, 3y를 사용한다."""
        mock_download.return_value = sample_ohlcv_df
        fetcher.fetch()
        mock_download.assert_called_once_with(
            "^GSPC", period="3y", auto_adjust=True, progress=False
        )

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_fetch_includes_volume(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """Volume 컬럼이 포함되어 수집된다."""
        mock_download.return_value = sample_ohlcv_df
        result = fetcher.fetch()
        assert "Volume" in result.columns
        assert result["Volume"].notna().all()


# ---------------------------------------------------------------------------
# MultiIndex 컬럼 처리 테스트
# ---------------------------------------------------------------------------


class TestMultiIndexHandling:
    """yfinance MultiIndex 컬럼 반환 시 처리 검증."""

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_handles_multiindex_columns(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """MultiIndex 컬럼을 단일 레벨로 변환한다."""
        # yfinance가 MultiIndex를 반환하는 경우 시뮬레이션
        multi_df = sample_ohlcv_df.copy()
        multi_df.columns = pd.MultiIndex.from_tuples(
            [(c, "^GSPC") for c in sample_ohlcv_df.columns]
        )
        mock_download.return_value = multi_df
        result = fetcher.fetch()
        assert not isinstance(result.columns, pd.MultiIndex)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# 결측값 처리 테스트
# ---------------------------------------------------------------------------


class TestForwardFill:
    """결측값 전방 채움(forward fill) 처리 검증."""

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_forward_fill_removes_nans(
        self, mock_download: MagicMock, fetcher: DataFetcher, ohlcv_with_nans: pd.DataFrame
    ):
        """전방 채움 후 첫 번째 유효값 이후에 NaN이 없다."""
        mock_download.return_value = ohlcv_with_nans
        result = fetcher.fetch()
        # 첫 행은 유효하므로 이후 모든 값에 NaN이 없어야 함
        assert result.iloc[1:].notna().all().all()

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_forward_fill_preserves_valid_values(
        self, mock_download: MagicMock, fetcher: DataFetcher, sample_ohlcv_df: pd.DataFrame
    ):
        """NaN이 없는 데이터는 전방 채움 후에도 값이 변경되지 않는다."""
        mock_download.return_value = sample_ohlcv_df
        result = fetcher.fetch()
        pd.testing.assert_frame_equal(result, sample_ohlcv_df)

    def test_validate_and_clean_fills_nans(
        self, fetcher: DataFetcher, ohlcv_with_nans: pd.DataFrame
    ):
        """_validate_and_clean이 NaN을 전방 채움한다."""
        result = fetcher._validate_and_clean(ohlcv_with_nans)
        # 첫 행 이후 NaN 없음
        assert result.iloc[1:].notna().all().all()


# ---------------------------------------------------------------------------
# 콘솔 출력 테스트
# ---------------------------------------------------------------------------


class TestConsolePrint:
    """수집 데이터 요약 콘솔 출력 검증."""

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_prints_summary(
        self,
        mock_download: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
        capsys: pytest.CaptureFixture[str],
    ):
        """시작일, 종료일, 총 거래일 수를 콘솔에 출력한다."""
        mock_download.return_value = sample_ohlcv_df
        fetcher.fetch()
        captured = capsys.readouterr()
        assert "시작일" in captured.out
        assert "종료일" in captured.out
        assert "총 거래일 수" in captured.out

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_prints_correct_dates(
        self,
        mock_download: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
        capsys: pytest.CaptureFixture[str],
    ):
        """출력된 시작일/종료일이 실제 데이터와 일치한다."""
        mock_download.return_value = sample_ohlcv_df
        fetcher.fetch()
        captured = capsys.readouterr()
        expected_start = sample_ohlcv_df.index[0].strftime("%Y-%m-%d")
        expected_end = sample_ohlcv_df.index[-1].strftime("%Y-%m-%d")
        assert expected_start in captured.out
        assert expected_end in captured.out

    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_prints_correct_trading_days(
        self,
        mock_download: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
        capsys: pytest.CaptureFixture[str],
    ):
        """출력된 총 거래일 수가 실제 행 수와 일치한다."""
        mock_download.return_value = sample_ohlcv_df
        fetcher.fetch()
        captured = capsys.readouterr()
        assert f"{len(sample_ohlcv_df)}일" in captured.out


# ---------------------------------------------------------------------------
# 재시도 및 에러 처리 테스트
# ---------------------------------------------------------------------------


class TestRetryAndErrorHandling:
    """네트워크 오류 재시도 및 에러 처리 검증."""

    @patch("sp500_backtest.data.fetcher.time.sleep")
    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_retries_on_network_error(
        self,
        mock_download: MagicMock,
        mock_sleep: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """네트워크 오류 후 재시도하여 성공한다."""
        mock_download.side_effect = [
            ConnectionError("네트워크 오류"),
            sample_ohlcv_df,
        ]
        result = fetcher.fetch()
        assert isinstance(result, pd.DataFrame)
        assert mock_download.call_count == 2

    @patch("sp500_backtest.data.fetcher.time.sleep")
    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_exits_after_max_retries(
        self,
        mock_download: MagicMock,
        mock_sleep: MagicMock,
        fetcher: DataFetcher,
    ):
        """MAX_RETRIES회 실패 후 sys.exit(1)로 종료한다."""
        mock_download.side_effect = ConnectionError("네트워크 오류")
        with pytest.raises(SystemExit) as exc_info:
            fetcher.fetch()
        assert exc_info.value.code == 1
        assert mock_download.call_count == DataFetcher.MAX_RETRIES

    @patch("sp500_backtest.data.fetcher.time.sleep")
    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_prints_error_on_max_retries(
        self,
        mock_download: MagicMock,
        mock_sleep: MagicMock,
        fetcher: DataFetcher,
        capsys: pytest.CaptureFixture[str],
    ):
        """MAX_RETRIES회 실패 시 에러 메시지를 출력한다."""
        mock_download.side_effect = ConnectionError("네트워크 오류")
        with pytest.raises(SystemExit):
            fetcher.fetch()
        captured = capsys.readouterr()
        assert "재시도" in captured.out
        assert "실패" in captured.out

    @patch("sp500_backtest.data.fetcher.time.sleep")
    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_retries_on_empty_dataframe(
        self,
        mock_download: MagicMock,
        mock_sleep: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """빈 DataFrame 반환 시 재시도한다."""
        mock_download.side_effect = [
            pd.DataFrame(),
            sample_ohlcv_df,
        ]
        result = fetcher.fetch()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_df)

    @patch("sp500_backtest.data.fetcher.time.sleep")
    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_exits_on_persistent_empty_data(
        self,
        mock_download: MagicMock,
        mock_sleep: MagicMock,
        fetcher: DataFetcher,
    ):
        """빈 DataFrame이 계속 반환되면 sys.exit(1)로 종료한다."""
        mock_download.return_value = pd.DataFrame()
        with pytest.raises(SystemExit) as exc_info:
            fetcher.fetch()
        assert exc_info.value.code == 1

    @patch("sp500_backtest.data.fetcher.time.sleep")
    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_retry_delay_between_attempts(
        self,
        mock_download: MagicMock,
        mock_sleep: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """재시도 사이에 대기 시간이 적용된다."""
        mock_download.side_effect = [
            ConnectionError("오류"),
            sample_ohlcv_df,
        ]
        fetcher.fetch()
        mock_sleep.assert_called_once_with(DataFetcher.RETRY_DELAY)

    @patch("sp500_backtest.data.fetcher.time.sleep")
    @patch("sp500_backtest.data.fetcher.yf.download")
    def test_succeeds_on_third_attempt(
        self,
        mock_download: MagicMock,
        mock_sleep: MagicMock,
        fetcher: DataFetcher,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """3번째 시도에서 성공한다."""
        mock_download.side_effect = [
            ConnectionError("오류 1"),
            ConnectionError("오류 2"),
            sample_ohlcv_df,
        ]
        result = fetcher.fetch()
        assert isinstance(result, pd.DataFrame)
        assert mock_download.call_count == 3
