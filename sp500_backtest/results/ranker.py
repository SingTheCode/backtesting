"""결과 정렬 및 순위 산출 모듈.

백테스팅 결과를 지정된 기준(총 수익률, CAGR, 샤프 비율)으로 내림차순 정렬하고,
CSV 저장 및 상위 N개 조합 요약 콘솔 출력 기능을 제공한다.
"""

import os

import pandas as pd

from sp500_backtest.engine.backtest import BacktestResult

# 결과 테이블 필수 컬럼 목록
REQUIRED_COLUMNS = [
    "combination_id",
    "total_return",
    "cagr",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "total_trades",
    "win_rate",
]

# 정렬 기준으로 허용되는 컬럼
VALID_SORT_COLUMNS = {"total_return", "cagr", "sharpe_ratio"}


class ResultRanker:
    """결과 정렬 및 순위 산출.

    BacktestResult 리스트를 DataFrame으로 변환하고,
    지정된 기준으로 내림차순 정렬하여 순위를 매긴다.
    """

    def rank(
        self,
        results: list[BacktestResult],
        sort_by: str = "total_return",
    ) -> pd.DataFrame:
        """결과를 지정된 기준으로 내림차순 정렬한다.

        Args:
            results: 백테스팅 결과 리스트.
            sort_by: 정렬 기준 컬럼 ("total_return", "cagr", "sharpe_ratio").

        Returns:
            정렬된 DataFrame (컬럼: combination_id, total_return, cagr,
            max_drawdown, sharpe_ratio, sortino_ratio, total_trades, win_rate).

        Raises:
            ValueError: sort_by가 허용되지 않는 컬럼인 경우.
        """
        if sort_by not in VALID_SORT_COLUMNS:
            raise ValueError(
                f"sort_by는 {VALID_SORT_COLUMNS} 중 하나여야 합니다: {sort_by}"
            )

        if not results:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        rows = [
            {
                "combination_id": r.combination_id,
                "total_return": r.total_return,
                "cagr": r.cagr,
                "max_drawdown": r.max_drawdown,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
            }
            for r in results
        ]

        df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        df = df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        return df

    def save_csv(self, ranked_df: pd.DataFrame, path: str) -> None:
        """결과 DataFrame을 CSV 파일로 저장한다.

        출력 디렉토리가 존재하지 않으면 자동 생성한다.

        Args:
            ranked_df: 정렬된 결과 DataFrame.
            path: 저장할 CSV 파일 경로.
        """
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        ranked_df.to_csv(path, index=False)

    def print_summary(self, ranked_df: pd.DataFrame, top_n: int = 20) -> None:
        """상위 N개 조합 요약을 콘솔에 출력한다.

        Args:
            ranked_df: 정렬된 결과 DataFrame.
            top_n: 출력할 상위 조합 수 (기본값: 20).
        """
        top_df = ranked_df.head(top_n).copy()

        if top_df.empty:
            print("결과가 없습니다.")
            return

        # 순위 컬럼 추가 (1부터 시작)
        top_df.insert(0, "rank", range(1, len(top_df) + 1))

        print(f"\n{'=' * 80}")
        print(f"  상위 {min(top_n, len(ranked_df))}개 조합 요약")
        print(f"{'=' * 80}")
        print(top_df.to_string(index=False))
        print(f"{'=' * 80}\n")
