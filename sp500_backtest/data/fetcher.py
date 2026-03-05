"""S&P 500 OHLCV 데이터 수집 모듈.

yfinance를 사용하여 S&P 500(^GSPC) 일봉 OHLCV 데이터를 수집하고,
결측값 전방 채움(forward fill) 처리를 수행한다.
네트워크 오류 시 3회 재시도 후 안전 종료한다.
"""

import sys
import time

import pandas as pd
import yfinance as yf


class DataFetcher:
    """S&P 500 OHLCV 데이터 수집기.

    yfinance를 사용하여 지정된 심볼의 일봉 OHLCV 데이터를 수집한다.
    네트워크 오류 시 최대 3회 재시도하며, 결측값은 전방 채움으로 처리한다.
    """

    MAX_RETRIES = 3  # 최대 재시도 횟수
    RETRY_DELAY = 2  # 재시도 간 대기 시간 (초)
    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]  # 필수 OHLCV 컬럼

    def fetch(self, symbol: str = "^GSPC", period: str = "3y") -> pd.DataFrame:
        """yfinance를 사용하여 OHLCV 데이터를 수집한다.

        Args:
            symbol: 수집 대상 심볼 (기본값: '^GSPC').
            period: 수집 기간 (기본값: '3y').

        Returns:
            pd.DataFrame: columns=['Open','High','Low','Close','Volume'], index=DatetimeIndex.
        """
        df = self._download_with_retry(symbol, period)
        df = self._validate_and_clean(df)
        self._print_summary(df)
        return df

    def _download_with_retry(self, symbol: str, period: str) -> pd.DataFrame:
        """네트워크 오류 시 최대 MAX_RETRIES회 재시도하여 데이터를 다운로드한다.

        Args:
            symbol: 수집 대상 심볼.
            period: 수집 기간.

        Returns:
            pd.DataFrame: 다운로드된 원시 OHLCV 데이터.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                df = yf.download(symbol, period=period, auto_adjust=True, progress=False)

                if df is None or df.empty:
                    raise ValueError(f"'{symbol}' 심볼에 대한 데이터가 비어 있습니다.")

                # MultiIndex 컬럼 처리 (yfinance가 단일 심볼에도 MultiIndex를 반환할 수 있음)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 필수 컬럼 존재 확인
                missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
                if missing:
                    raise ValueError(f"필수 컬럼이 누락되었습니다: {missing}")

                return df[self.REQUIRED_COLUMNS]

            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    print(
                        f"[DataFetcher] 데이터 수집 실패 (시도 {attempt}/{self.MAX_RETRIES}): {e}"
                    )
                    time.sleep(self.RETRY_DELAY)

        print(f"[DataFetcher] 오류: {self.MAX_RETRIES}회 재시도 후에도 데이터 수집에 실패했습니다.")
        print(f"[DataFetcher] 마지막 오류: {last_error}")
        sys.exit(1)

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측값 전방 채움(forward fill) 처리.

        Args:
            df: 원시 OHLCV DataFrame.

        Returns:
            pd.DataFrame: 결측값이 전방 채움 처리된 DataFrame.
        """
        df = df.ffill()
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """수집된 데이터의 시작일, 종료일, 총 거래일 수를 콘솔에 출력한다.

        Args:
            df: 정제된 OHLCV DataFrame.
        """
        start_date = df.index[0].strftime("%Y-%m-%d")  # 시작일
        end_date = df.index[-1].strftime("%Y-%m-%d")  # 종료일
        total_days = len(df)  # 총 거래일 수

        print(f"[DataFetcher] 데이터 수집 완료")
        print(f"  시작일: {start_date}")
        print(f"  종료일: {end_date}")
        print(f"  총 거래일 수: {total_days}일")
