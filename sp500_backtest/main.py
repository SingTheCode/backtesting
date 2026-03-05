"""메인 오케스트레이터 모듈.

config.yaml 로딩 → DataFetcher → CombinationEngine → ParameterOptimizer →
SignalGenerator → BacktestEngine → ResultRanker → ReportGenerator 순서로
전체 백테스팅 파이프라인을 연결한다.

IndicatorCache를 파이프라인 전체에서 공유하여 중복 계산을 방지하고,
체크포인트 저장/복원 및 진행률/ETA 콘솔 출력을 통합한다.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd

from sp500_backtest.config import load_config
from sp500_backtest.data.fetcher import DataFetcher
from sp500_backtest.engine.backtest import BacktestEngine, BacktestResult
from sp500_backtest.engine.cache import IndicatorCache
from sp500_backtest.engine.checkpoint import (
    Checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from sp500_backtest.engine.combination import CombinationEngine, IndicatorCombination
from sp500_backtest.engine.optimizer import (
    ParameterOptimizer,
    ParamSearchSpace,
    _build_confirmation_registry,
    _build_leading_registry,
    _run_single_backtest,
)
from sp500_backtest.engine.signal import SignalGenerator
from sp500_backtest.indicators.base import ConfirmationIndicator, IndicatorResult
from sp500_backtest.results.ranker import ResultRanker
from sp500_backtest.results.reporter import ReportGenerator

# 체크포인트 기본 저장 경로
_DEFAULT_CHECKPOINT_PATH = "sp500_backtest/checkpoint/pipeline.pkl"


def build_leading_names() -> list[str]:
    """리딩 지표 레지스트리에서 모든 지표 이름 목록을 추출한다.

    Returns:
        리딩 지표 이름 리스트 (예: ["Range Filter", "RQK", ...]).
    """
    registry = _build_leading_registry()
    return list(registry.keys())


def build_confirmation_info() -> list[dict[str, Any]]:
    """확인 지표 레지스트리에서 이름과 서브타입 정보를 추출한다.

    Returns:
        확인 지표 정보 리스트.
        각 항목: {"name": str, "subtypes": list[str]}.
    """
    registry = _build_confirmation_registry()
    info: list[dict[str, Any]] = []
    for name, cls in registry.items():
        instance = cls()
        info.append({
            "name": name,
            "subtypes": instance.subtypes,
        })
    return info


def _build_search_spaces(
    param_ranges: dict[str, dict[str, dict[str, float]]],
) -> dict[str, list[ParamSearchSpace]]:
    """config의 param_ranges를 ParameterOptimizer용 search_spaces로 변환한다.

    Args:
        param_ranges: config.yaml의 param_ranges 섹션.
            예: {"ema_cross": {"fast_period": {"min": 5, "max": 50, "step": 5}}}

    Returns:
        지표 이름 → ParamSearchSpace 리스트 매핑.
    """
    search_spaces: dict[str, list[ParamSearchSpace]] = {}
    for indicator_key, params in param_ranges.items():
        spaces: list[ParamSearchSpace] = []
        for param_name, range_def in params.items():
            spaces.append(
                ParamSearchSpace(
                    name=param_name,
                    min_val=float(range_def["min"]),
                    max_val=float(range_def["max"]),
                    step=float(range_def["step"]),
                )
            )
        search_spaces[indicator_key] = spaces
    return search_spaces


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


def _run_combination_backtest(
    combination: IndicatorCombination,
    df: pd.DataFrame,
    config: dict[str, Any],
    search_spaces: dict[str, list[ParamSearchSpace]],
    cache: IndicatorCache,
) -> list[BacktestResult]:
    """단일 조합에 대해 백테스팅을 수행한다.

    param_ranges에 해당 지표의 탐색 범위가 있으면 ParameterOptimizer를 사용하고,
    없으면 기본 파라미터로 단일 백테스팅을 수행한다.

    Args:
        combination: 지표 조합 정의.
        df: OHLCV DataFrame.
        config: 전체 설정 딕셔너리.
        search_spaces: 파라미터 탐색 공간 매핑.
        cache: 공유 IndicatorCache 인스턴스.

    Returns:
        BacktestResult 리스트 (최적화 시 여러 개, 단일 백테스팅 시 0~1개).
    """
    backtest_cfg = config.get("backtest", {})
    signal_expiry = backtest_cfg.get("signal_expiry", 3)  # 시그널 만료 캔들 수
    alternate_signal = backtest_cfg.get("alternate_signal", True)  # 연속 동일 방향 필터링
    transaction_cost = backtest_cfg.get("transaction_cost", 0.001)  # 거래 비용 비율

    # 해당 조합의 리딩 지표에 대한 파라미터 탐색 범위 확인
    combo_search: dict[str, list[ParamSearchSpace]] = {}
    for key, spaces in search_spaces.items():
        # 지표 이름과 param_ranges 키를 매칭 (소문자/언더스코어 정규화)
        normalized_key = key.lower().replace(" ", "_")
        normalized_leading = combination.leading.lower().replace(" ", "_")
        if normalized_key == normalized_leading:
            combo_search[combination.leading] = spaces

    if combo_search:
        # 파라미터 최적화 수행
        optimizer_cfg = config.get("optimizer", {})
        perf_cfg = config.get("performance", {})
        optimizer = ParameterOptimizer(
            method=optimizer_cfg.get("method", "grid"),
            n_workers=perf_cfg.get("n_workers", -1),
        )
        return optimizer.optimize(
            combination=combination,
            df=df,
            search_spaces=combo_search,
            signal_expiry=signal_expiry,
            alternate_signal=alternate_signal,
            transaction_cost=transaction_cost,
            random_iterations=optimizer_cfg.get("random_iterations", 1000),
        )
    else:
        # 기본 파라미터로 단일 백테스팅
        result = _run_single_backtest(
            combination=combination,
            df=df,
            param_set={},
            signal_expiry=signal_expiry,
            alternate_signal=alternate_signal,
            transaction_cost=transaction_cost,
        )
        return [result] if result is not None else []


def run_pipeline(config: dict[str, Any] | None = None) -> list[BacktestResult]:
    """전체 백테스팅 파이프라인을 실행한다.

    1. 설정 로딩
    2. OHLCV 데이터 수집
    3. 지표 레지스트리 구축
    4. 조합 생성
    5. 체크포인트 복원 (있으면 이어서 실행)
    6. 각 조합별 백테스팅 (진행률/ETA 출력)
    7. 결과 정렬 및 CSV 저장
    8. QuantStats 리포트 생성
    9. 요약 출력

    Args:
        config: 설정 딕셔너리. None이면 config.yaml에서 로딩.

    Returns:
        전체 BacktestResult 리스트.
    """
    # ── 1. 설정 로딩 ──
    if config is None:
        config = load_config()
    print("=" * 60)
    print("  S&P 500 백테스팅 파이프라인 시작")
    print("=" * 60)

    # ── 2. 데이터 수집 ──
    data_cfg = config.get("data", {})
    fetcher = DataFetcher()
    df = fetcher.fetch(
        symbol=data_cfg.get("symbol", "^GSPC"),
        period=data_cfg.get("period", "3y"),
    )

    # 벤치마크 수익률 (Buy & Hold S&P 500)
    benchmark_returns = df["Close"].pct_change().fillna(0.0)

    # ── 3. 지표 레지스트리 구축 ──
    print("\n지표 레지스트리 구축 중...")
    leading_names = build_leading_names()
    confirmation_info = build_confirmation_info()
    print(f"  리딩 지표: {len(leading_names)}개")
    print(f"  확인 지표: {len(confirmation_info)}개")

    # ── 4. 조합 생성 ──
    combo_cfg = config.get("combination", {})
    engine = CombinationEngine(
        leading_names=leading_names,
        confirmation_info=confirmation_info,
    )
    combinations = engine.generate(
        max_confirmations=combo_cfg.get("max_confirmations", 3),
        max_combinations=combo_cfg.get("max_combinations", 100_000),
    )

    # ── 5. 체크포인트 복원 ──
    perf_cfg = config.get("performance", {})
    checkpoint_interval = perf_cfg.get("checkpoint_interval", 100)  # 체크포인트 저장 간격
    checkpoint = load_checkpoint(_DEFAULT_CHECKPOINT_PATH)

    all_results: list[BacktestResult] = []
    completed_ids: set[str] = set()

    if checkpoint is not None:
        all_results = list(checkpoint.results)
        completed_ids = set(checkpoint.completed_combinations)
        print(f"\n체크포인트 복원: {len(completed_ids)}/{len(combinations)} 조합 완료됨")

    # ── 6. 조합별 백테스팅 ──
    search_spaces = _build_search_spaces(config.get("param_ranges", {}))
    cache = IndicatorCache()  # 파이프라인 전체 공유 캐시

    remaining = [c for c in combinations if c.id not in completed_ids]
    total_remaining = len(remaining)
    total_all = len(combinations)

    if total_remaining > 0:
        print(f"\n백테스팅 시작: {total_remaining}개 조합 (전체 {total_all}개)")
        start_time = time.time()

        for i, combo in enumerate(remaining, start=1):
            results = _run_combination_backtest(
                combination=combo,
                df=df,
                config=config,
                search_spaces=search_spaces,
                cache=cache,
            )
            all_results.extend(results)
            completed_ids.add(combo.id)

            # 진행률 및 ETA 출력
            elapsed = time.time() - start_time
            if i > 0:
                eta_seconds = (elapsed / i) * (total_remaining - i)
            else:
                eta_seconds = 0.0
            pct = i / total_remaining * 100

            # 10% 간격 또는 마지막 조합에서 출력
            if i % max(1, total_remaining // 10) == 0 or i == total_remaining:
                print(
                    f"  파이프라인 진행: {i}/{total_remaining} ({pct:.1f}%) | "
                    f"ETA: {_format_eta(eta_seconds)} | "
                    f"캐시 히트: {cache.hits}"
                )

            # 체크포인트 저장 (checkpoint_interval 간격)
            if i % checkpoint_interval == 0:
                cp = Checkpoint(
                    completed_combinations=list(completed_ids),
                    results=all_results,
                    total_combinations=total_all,
                )
                save_checkpoint(cp, _DEFAULT_CHECKPOINT_PATH)

        # 최종 체크포인트 저장
        cp = Checkpoint(
            completed_combinations=list(completed_ids),
            results=all_results,
            total_combinations=total_all,
        )
        save_checkpoint(cp, _DEFAULT_CHECKPOINT_PATH)
        print(f"\n백테스팅 완료: 총 {len(all_results)}개 결과")
        print(f"  캐시 통계 — 히트: {cache.hits}, 미스: {cache.misses}, 크기: {cache.size}")
    else:
        print("\n모든 조합이 이미 완료되었습니다.")

    # ── 7. 결과 정렬 및 CSV 저장 ──
    results_cfg = config.get("results", {})
    sort_by = results_cfg.get("sort_by", "total_return")  # 정렬 기준
    top_n_display = results_cfg.get("top_n_display", 20)  # 콘솔 출력 상위 N개
    top_n_report = results_cfg.get("top_n_report", 5)  # 리포트 생성 상위 N개

    ranker = ResultRanker()
    ranked_df = ranker.rank(all_results, sort_by=sort_by)
    ranker.save_csv(ranked_df, "sp500_backtest/output/results.csv")
    ranker.print_summary(ranked_df, top_n=top_n_display)

    # ── 8. QuantStats 리포트 생성 ──
    if all_results:
        # total_return 기준 상위 N개 결과 추출
        sorted_results = sorted(all_results, key=lambda r: r.total_return, reverse=True)
        top_results = sorted_results[:top_n_report]

        reporter = ReportGenerator()
        reporter.generate(
            results=top_results,
            benchmark_returns=benchmark_returns,
            top_n=top_n_report,
            output_dir="sp500_backtest/output/reports",
        )

    # ── 9. 완료 ──
    print("\n" + "=" * 60)
    print("  파이프라인 완료")
    print("=" * 60)

    return all_results


def main() -> None:
    """메인 엔트리포인트. 명령줄 인자 없이 기본값으로 실행한다."""
    run_pipeline()


if __name__ == "__main__":
    main()
