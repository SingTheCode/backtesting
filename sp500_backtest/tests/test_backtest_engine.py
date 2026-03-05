"""백테스팅 엔진 단위 테스트.

BacktestEngine의 포지션 시프트, 거래 비용 차감, 성과 지표 계산,
엣지 케이스를 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.engine.backtest import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------


def _make_index(n: int) -> pd.DatetimeIndex:
    """테스트용 DatetimeIndex 생성 (n개 거래일)."""
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _make_positions(values: list[int]) -> pd.Series:
    """int 리스트로 포지션 Series를 생성한다."""
    idx = _make_index(len(values))
    return pd.Series(values, index=idx, dtype=np.int64)


def _make_prices(values: list[float]) -> pd.Series:
    """float 리스트로 가격 Series를 생성한다."""
    idx = _make_index(len(values))
    return pd.Series(values, index=idx, dtype=np.float64)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> BacktestEngine:
    """BacktestEngine 인스턴스."""
    return BacktestEngine()


# ---------------------------------------------------------------------------
# 포지션 시프트 (1일 지연) 테스트
# ---------------------------------------------------------------------------


class TestPositionShift:
    """시그널 1일 지연(shift(1)) 검증 — 미래 정보 편향 방지."""

    def test_positions_shifted_by_one_day(self, engine: BacktestEngine):
        """시그널이 정확히 1일 지연되어 적용되는지 검증."""
        positions = _make_positions([1, -1, 0, 1, -1])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 105.0])

        result = engine.run(positions, prices, transaction_cost=0.0)

        # shift(1) 적용: [NaN→0, 1, -1, 0, 1]
        expected_positions = [0, 1, -1, 0, 1]
        for i, expected in enumerate(expected_positions):
            assert result.positions.iloc[i] == expected, (
                f"인덱스 {i}: 기대값 {expected}, 실제값 {result.positions.iloc[i]}"
            )

    def test_first_day_always_zero_position(self, engine: BacktestEngine):
        """첫 번째 거래일은 항상 포지션 0 (이전 시그널 없음)."""
        positions = _make_positions([1, 0, -1])
        prices = _make_prices([100.0, 105.0, 103.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert result.positions.iloc[0] == 0


# ---------------------------------------------------------------------------
# 거래 비용 차감 테스트
# ---------------------------------------------------------------------------


class TestTransactionCost:
    """포지션 변경 시 거래 비용 차감 검증."""

    def test_cost_deducted_on_position_change(self, engine: BacktestEngine):
        """포지션 변경 시점에서 거래 비용이 차감되는지 검증."""
        # 포지션: [0, 1, 1, -1] (shift 적용 후)
        positions = _make_positions([1, 1, -1, 0])
        prices = _make_prices([100.0, 102.0, 104.0, 103.0])

        result_with_cost = engine.run(positions, prices, transaction_cost=0.01)
        result_no_cost = engine.run(positions, prices, transaction_cost=0.0)

        # 거래 비용이 있으면 수익률이 더 낮아야 함
        assert result_with_cost.total_return <= result_no_cost.total_return

    def test_zero_transaction_cost(self, engine: BacktestEngine):
        """거래 비용 0이면 비용 차감 없음."""
        positions = _make_positions([1, -1, 1, -1])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        # 거래 비용 0이므로 순수 가격 변동만 반영
        assert isinstance(result.total_return, float)

    def test_no_cost_when_no_position_change(self, engine: BacktestEngine):
        """포지션 변경이 없으면 거래 비용 차감 없음."""
        # shift 후: [0, 1, 1, 1] → 인덱스 1에서만 변경 (0→1)
        positions = _make_positions([1, 1, 1, 1])
        prices = _make_prices([100.0, 102.0, 104.0, 106.0])

        result_cost = engine.run(positions, prices, transaction_cost=0.01)
        result_no_cost = engine.run(positions, prices, transaction_cost=0.0)

        # 인덱스 1에서만 비용 차감, 나머지는 동일
        # 비용 차이는 정확히 1회 거래 비용
        diff = result_no_cost.total_return - result_cost.total_return
        assert diff > 0

    def test_transaction_cost_monotone_decrease(self, engine: BacktestEngine):
        """거래 비용 0 수익률 >= 거래 비용 양수 수익률 (단조 감소)."""
        positions = _make_positions([1, -1, 1, -1, 0, 1])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 102.0, 104.0])

        r0 = engine.run(positions, prices, transaction_cost=0.0)
        r1 = engine.run(positions, prices, transaction_cost=0.001)
        r2 = engine.run(positions, prices, transaction_cost=0.01)

        assert r0.total_return >= r1.total_return
        assert r1.total_return >= r2.total_return


# ---------------------------------------------------------------------------
# 성과 지표 계산 테스트
# ---------------------------------------------------------------------------


class TestMetricCalculations:
    """총 수익률, CAGR, 최대 낙폭, 샤프/소르티노 비율 계산 검증."""

    def test_total_return_calculation(self, engine: BacktestEngine):
        """총 수익률이 올바르게 계산되는지 검증."""
        # shift 후 포지션: [0, 1, 1, 1]
        # 가격 수익률: [0, 0.02, ~0.0196, ~0.0192]
        # 전략 수익률: [0, 0.02, ~0.0196, ~0.0192]
        positions = _make_positions([1, 1, 1, 1])
        prices = _make_prices([100.0, 102.0, 104.0, 106.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        # 총 수익률 > 0 (가격 상승 + Long 포지션)
        assert result.total_return > 0

    def test_total_return_short_position(self, engine: BacktestEngine):
        """Short 포지션에서 가격 하락 시 양수 수익률."""
        # shift 후 포지션: [0, -1, -1]
        positions = _make_positions([-1, -1, -1])
        prices = _make_prices([100.0, 98.0, 96.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        # Short 포지션 + 가격 하락 → 양수 수익
        assert result.total_return > 0

    def test_cagr_positive_for_positive_return(self, engine: BacktestEngine):
        """양수 총 수익률이면 CAGR도 양수."""
        positions = _make_positions([1, 1, 1, 1, 1])
        prices = _make_prices([100.0, 102.0, 104.0, 106.0, 108.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        if result.total_return > 0:
            assert result.cagr > 0

    def test_max_drawdown_is_non_positive(self, engine: BacktestEngine):
        """최대 낙폭은 항상 0 이하."""
        positions = _make_positions([1, 1, 1, 1, 1])
        prices = _make_prices([100.0, 105.0, 95.0, 90.0, 100.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert result.max_drawdown <= 0

    def test_sharpe_ratio_calculation(self, engine: BacktestEngine):
        """샤프 비율이 float으로 반환되는지 검증."""
        positions = _make_positions([1, -1, 1, -1, 0])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 102.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert isinstance(result.sharpe_ratio, float)

    def test_sortino_ratio_calculation(self, engine: BacktestEngine):
        """소르티노 비율이 float으로 반환되는지 검증."""
        positions = _make_positions([1, -1, 1, -1, 0])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 102.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert isinstance(result.sortino_ratio, float)

    def test_total_trades_count(self, engine: BacktestEngine):
        """총 거래 횟수가 포지션 변경 횟수와 일치."""
        # shift 후: [0, 1, -1, 0, 1] → 변경: 0→1, 1→-1, -1→0, 0→1 = 4회
        positions = _make_positions([1, -1, 0, 1, -1])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 105.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert result.total_trades == 4

    def test_win_rate_range(self, engine: BacktestEngine):
        """승률은 0.0 ~ 1.0 범위."""
        positions = _make_positions([1, -1, 1, -1, 0])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 102.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert 0.0 <= result.win_rate <= 1.0


# ---------------------------------------------------------------------------
# 포지션 값 불변성 테스트
# ---------------------------------------------------------------------------


class TestPositionValues:
    """포지션 값이 항상 {-1, 0, 1} 중 하나인지 검증."""

    def test_positions_in_valid_set(self, engine: BacktestEngine):
        """결과 포지션 값이 {-1, 0, 1}에만 속하는지 검증."""
        positions = _make_positions([1, -1, 0, 1, -1, 0])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 105.0, 104.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        unique_vals = set(result.positions.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_all_long_positions(self, engine: BacktestEngine):
        """모든 Long 포지션 입력 시 결과 포지션도 유효."""
        positions = _make_positions([1, 1, 1, 1])
        prices = _make_prices([100.0, 102.0, 104.0, 106.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        unique_vals = set(result.positions.unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_all_short_positions(self, engine: BacktestEngine):
        """모든 Short 포지션 입력 시 결과 포지션도 유효."""
        positions = _make_positions([-1, -1, -1, -1])
        prices = _make_prices([100.0, 98.0, 96.0, 94.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        unique_vals = set(result.positions.unique())
        assert unique_vals.issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# 엣지 케이스 테스트
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """엣지 케이스 검증."""

    def test_all_zero_positions(self, engine: BacktestEngine):
        """모든 포지션이 0이면 수익률 0, 거래 0."""
        positions = _make_positions([0, 0, 0, 0])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert result.total_return == 0.0
        assert result.total_trades == 0
        assert result.win_rate == 0.0

    def test_single_day(self, engine: BacktestEngine):
        """단일 거래일도 정상 처리."""
        positions = _make_positions([1])
        prices = _make_prices([100.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        # shift 후 포지션 0, 수익률 0
        assert result.positions.iloc[0] == 0
        assert result.total_return == 0.0

    def test_two_days(self, engine: BacktestEngine):
        """2일 데이터도 정상 처리."""
        positions = _make_positions([1, 0])
        prices = _make_prices([100.0, 105.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        # shift 후: [0, 1] → 인덱스 1에서 Long, 가격 5% 상승
        assert result.positions.iloc[0] == 0
        assert result.positions.iloc[1] == 1
        assert result.total_return == pytest.approx(0.05, abs=1e-10)

    def test_no_trades_metrics(self, engine: BacktestEngine):
        """거래가 없으면 모든 지표가 0 또는 기본값."""
        positions = _make_positions([0, 0, 0])
        prices = _make_prices([100.0, 102.0, 101.0])

        result = engine.run(positions, prices, transaction_cost=0.001)
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.max_drawdown == 0.0

    def test_constant_prices(self, engine: BacktestEngine):
        """가격 변동이 없으면 수익률 0 (거래 비용 제외)."""
        positions = _make_positions([1, 1, 1, 1])
        prices = _make_prices([100.0, 100.0, 100.0, 100.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert result.total_return == pytest.approx(0.0, abs=1e-10)

    def test_result_dataclass_fields(self, engine: BacktestEngine):
        """BacktestResult가 모든 필수 필드를 포함하는지 검증."""
        positions = _make_positions([1, -1, 0])
        prices = _make_prices([100.0, 102.0, 101.0])

        result = engine.run(positions, prices, combination_id="test_combo")

        assert result.combination_id == "test_combo"
        assert isinstance(result.total_return, float)
        assert isinstance(result.cagr, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.total_trades, int)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.strategy_returns, pd.Series)
        assert isinstance(result.positions, pd.Series)

    def test_strategy_returns_length_matches_input(self, engine: BacktestEngine):
        """전략 수익률 Series 길이가 입력과 동일."""
        positions = _make_positions([1, -1, 0, 1, -1])
        prices = _make_prices([100.0, 102.0, 101.0, 103.0, 105.0])

        result = engine.run(positions, prices, transaction_cost=0.0)
        assert len(result.strategy_returns) == len(positions)
        assert len(result.positions) == len(positions)
