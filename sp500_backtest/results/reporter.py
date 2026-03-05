"""QuantStats 성과 리포트 생성 모듈.

상위 N개 조합에 대해 QuantStats HTML 리포트를 생성하고,
S&P 500 Buy & Hold 벤치마크와 비교 분석을 수행한다.
핵심 지표(총 수익률, CAGR, Max Drawdown, Sharpe, Sortino)를 콘솔에 출력한다.
"""

import logging
import os
import re

import pandas as pd
import quantstats

from sp500_backtest.engine.backtest import BacktestResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """QuantStats 성과 리포트 생성기.

    상위 N개 조합에 대해 QuantStats HTML 리포트를 각각 생성하고,
    S&P 500 Buy & Hold 수익률을 벤치마크로 비교 분석한다.
    """

    def generate(
        self,
        results: list[BacktestResult],
        benchmark_returns: pd.Series,
        top_n: int = 5,
        output_dir: str = "output/reports",
    ) -> None:
        """상위 N개 조합에 대해 QuantStats HTML 리포트를 생성한다.

        Args:
            results: 백테스팅 결과 리스트 (정렬된 상태 또는 내부에서 정렬).
            benchmark_returns: S&P 500 Buy & Hold 일별 수익률 Series.
            top_n: 리포트를 생성할 상위 조합 수 (기본값: 5).
            output_dir: 리포트 출력 디렉토리 경로 (기본값: "output/reports").
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 상위 N개만 선택 (이미 정렬된 상태 가정, 아니면 total_return 기준 정렬)
        sorted_results = sorted(
            results, key=lambda r: r.total_return, reverse=True
        )
        top_results = sorted_results[:top_n]

        for rank, result in enumerate(top_results, start=1):
            filename = self._generate_filename(rank, result.combination_id)
            filepath = os.path.join(output_dir, filename)

            # 핵심 지표 콘솔 출력
            self._print_metrics(rank, result)

            try:
                quantstats.reports.html(
                    result.strategy_returns,
                    benchmark=benchmark_returns,
                    output=filepath,
                    title=f"Rank {rank:02d} - {result.combination_id}",
                )
            except Exception as e:
                logger.warning(
                    "리포트 생성 실패 (rank=%d, id=%s): %s",
                    rank,
                    result.combination_id,
                    e,
                )
                continue

    def _sanitize_filename(self, combination_id: str) -> str:
        """조합 ID를 파일명에 안전한 문자열로 변환한다.

        공백, 슬래시, 특수문자를 언더스코어로 치환하고,
        연속 언더스코어를 하나로 축소한 뒤 최대 100자로 잘라낸다.

        Args:
            combination_id: 원본 조합 식별자.

        Returns:
            파일명에 안전한 문자열.
        """
        # 영문, 숫자, 언더스코어, 하이픈 외 모든 문자를 언더스코어로 치환
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", combination_id)
        # 연속 언더스코어 축소
        sanitized = re.sub(r"_+", "_", sanitized)
        # 앞뒤 언더스코어 제거
        sanitized = sanitized.strip("_")
        # 최대 100자로 잘라냄
        return sanitized[:100]

    def _generate_filename(self, rank: int, combination_id: str) -> str:
        """순위와 조합 ID로 리포트 파일명을 생성한다.

        형식: rank{rank:02d}_{sanitized_id}.html

        Args:
            rank: 순위 (1부터 시작).
            combination_id: 조합 식별자.

        Returns:
            리포트 파일명 문자열.
        """
        sanitized = self._sanitize_filename(combination_id)
        return f"rank{rank:02d}_{sanitized}.html"

    @staticmethod
    def _print_metrics(rank: int, result: BacktestResult) -> None:
        """핵심 지표를 콘솔에 출력한다.

        Args:
            rank: 순위.
            result: 백테스팅 결과.
        """
        print(f"\n--- Rank {rank:02d}: {result.combination_id} ---")
        print(f"  총 수익률:     {result.total_return:>10.2%}")
        print(f"  CAGR:          {result.cagr:>10.2%}")
        print(f"  Max Drawdown:  {result.max_drawdown:>10.2%}")
        print(f"  Sharpe Ratio:  {result.sharpe_ratio:>10.4f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:>10.4f}")
