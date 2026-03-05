"""시그널 생성기 단위 테스트.

SignalGenerator의 AND 조건 결합, Signal Expiry, Alternate Signal 로직을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from sp500_backtest.engine.signal import SignalGenerator
from sp500_backtest.indicators.base import IndicatorResult


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------


def _make_index(n: int) -> pd.DatetimeIndex:
    """테스트용 DatetimeIndex 생성 (n개 거래일)."""
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _make_result(
    long_list: list[bool], short_list: list[bool]
) -> IndicatorResult:
    """bool 리스트로 IndicatorResult를 생성한다."""
    idx = _make_index(len(long_list))
    return IndicatorResult(
        long_signal=pd.Series(long_list, index=idx, dtype=bool),
        short_signal=pd.Series(short_list, index=idx, dtype=bool),
    )


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def gen() -> SignalGenerator:
    """SignalGenerator 인스턴스."""
    return SignalGenerator()


# ---------------------------------------------------------------------------
# AND 조건 결합 테스트
# ---------------------------------------------------------------------------


class TestANDCondition:
    """리딩 + 확인 지표 AND 조건 결합 검증."""

    def test_long_signal_all_confirm(self, gen: SignalGenerator):
        """리딩 Long + 모든 확인 Long → Long(+1) 시그널 생성."""
        leading = _make_result(
            [True, False, False, True, False],
            [False, False, False, False, False],
        )
        conf1 = _make_result(
            [True, True, False, True, False],
            [False, False, False, False, False],
        )
        conf2 = _make_result(
            [True, False, False, True, False],
            [False, False, False, False, False],
        )
        result = gen.generate(leading, [conf1, conf2], signal_expiry=3, alternate_signal=False)
        # 인덱스 0: 리딩 Long + 확인 모두 Long → 1
        assert result.iloc[0] == 1
        # 인덱스 3: 리딩 Long + 확인 모두 Long → 1
        assert result.iloc[3] == 1

    def test_short_signal_all_confirm(self, gen: SignalGenerator):
        """리딩 Short + 모든 확인 Short → Short(-1) 시그널 생성."""
        leading = _make_result(
            [False, False, True, False],
            [False, True, False, False],
        )
        conf1 = _make_result(
            [False, False, False, False],
            [False, True, False, False],
        )
        result = gen.generate(leading, [conf1], signal_expiry=3, alternate_signal=False)
        # 인덱스 1: 리딩 Short + 확인 Short → -1
        assert result.iloc[1] == -1

    def test_long_not_confirmed_stays_zero(self, gen: SignalGenerator):
        """리딩 Long이지만 확인 지표가 미확인이면 0 (expiry 내 확인 안 됨)."""
        leading = _make_result(
            [True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
        )
        # 확인 지표가 전혀 Long을 주지 않음
        conf1 = _make_result(
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False],
        )
        result = gen.generate(leading, [conf1], signal_expiry=3, alternate_signal=False)
        # 모든 시점에서 0이어야 함
        assert (result == 0).all()

    def test_partial_confirmation_stays_zero(self, gen: SignalGenerator):
        """확인 지표 중 일부만 확인하면 시그널 무효."""
        leading = _make_result(
            [True, False, False, False],
            [False, False, False, False],
        )
        conf1 = _make_result(
            [True, False, False, False],
            [False, False, False, False],
        )
        # conf2는 Long을 확인하지 않음
        conf2 = _make_result(
            [False, False, False, False],
            [False, False, False, False],
        )
        result = gen.generate(leading, [conf1, conf2], signal_expiry=3, alternate_signal=False)
        # conf2가 확인하지 않으므로 expiry 내에도 시그널 없음
        assert result.iloc[0] == 0


# ---------------------------------------------------------------------------
# 확인 지표 없는 경우 테스트
# ---------------------------------------------------------------------------


class TestNoConfirmations:
    """확인 지표가 없을 때 리딩 시그널 직접 사용 검증."""

    def test_leading_only_long(self, gen: SignalGenerator):
        """확인 지표 없으면 리딩 Long → Long(+1)."""
        leading = _make_result(
            [True, False, True, False],
            [False, False, False, False],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=False)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 0
        assert result.iloc[2] == 1

    def test_leading_only_short(self, gen: SignalGenerator):
        """확인 지표 없으면 리딩 Short → Short(-1)."""
        leading = _make_result(
            [False, False, False],
            [True, False, True],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=False)
        assert result.iloc[0] == -1
        assert result.iloc[1] == 0
        assert result.iloc[2] == -1

    def test_leading_both_long_and_short(self, gen: SignalGenerator):
        """리딩이 동시에 Long+Short이면 Short(-1)이 우선 (나중에 할당)."""
        leading = _make_result(
            [True, False],
            [True, False],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=False)
        # Short이 Long 이후에 할당되므로 -1
        assert result.iloc[0] == -1


# ---------------------------------------------------------------------------
# Signal Expiry 테스트
# ---------------------------------------------------------------------------


class TestSignalExpiry:
    """Signal Expiry 로직 검증."""

    def test_confirmation_within_expiry_window(self, gen: SignalGenerator):
        """리딩 시그널 후 expiry 윈도우 내 확인 → 유효 시그널."""
        # 리딩: 인덱스 0에서 Long 발생
        # 확인: 인덱스 2에서 Long 확인 (expiry=3 이내)
        leading = _make_result(
            [True, False, False, False, False],
            [False, False, False, False, False],
        )
        conf = _make_result(
            [False, False, True, False, False],
            [False, False, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=3, alternate_signal=False)
        # 인덱스 2에서 확인됨 (0+3=3 이내)
        assert result.iloc[2] == 1

    def test_confirmation_at_expiry_boundary(self, gen: SignalGenerator):
        """리딩 시그널 후 정확히 expiry 캔들에서 확인 → 유효."""
        # 리딩: 인덱스 0에서 Long
        # 확인: 인덱스 3에서 Long (expiry=3, 0+3=3 이내)
        leading = _make_result(
            [True, False, False, False, False],
            [False, False, False, False, False],
        )
        conf = _make_result(
            [False, False, False, True, False],
            [False, False, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=3, alternate_signal=False)
        assert result.iloc[3] == 1

    def test_confirmation_after_expiry_window(self, gen: SignalGenerator):
        """리딩 시그널 후 expiry 윈도우 초과 → 시그널 무효화."""
        # 리딩: 인덱스 0에서 Long
        # 확인: 인덱스 4에서 Long (expiry=3, 0+3=3 초과)
        leading = _make_result(
            [True, False, False, False, False, False],
            [False, False, False, False, False, False],
        )
        conf = _make_result(
            [False, False, False, False, True, False],
            [False, False, False, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=3, alternate_signal=False)
        # 인덱스 4는 expiry 초과이므로 0
        assert result.iloc[4] == 0

    def test_short_expiry_logic(self, gen: SignalGenerator):
        """Short 시그널에도 동일한 expiry 로직 적용."""
        leading = _make_result(
            [False, False, False, False, False],
            [True, False, False, False, False],
        )
        conf = _make_result(
            [False, False, False, False, False],
            [False, True, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=3, alternate_signal=False)
        # 인덱스 1에서 확인 (0+3=3 이내)
        assert result.iloc[1] == -1

    def test_expiry_one_candle(self, gen: SignalGenerator):
        """signal_expiry=1이면 바로 다음 캔들까지만 유효."""
        leading = _make_result(
            [True, False, False, False],
            [False, False, False, False],
        )
        conf = _make_result(
            [False, True, False, False],
            [False, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=1, alternate_signal=False)
        # 인덱스 1: 0+1=1 이내 → 유효
        assert result.iloc[1] == 1

    def test_expiry_one_candle_too_late(self, gen: SignalGenerator):
        """signal_expiry=1이면 2캔들 후 확인은 무효."""
        leading = _make_result(
            [True, False, False, False],
            [False, False, False, False],
        )
        conf = _make_result(
            [False, False, True, False],
            [False, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=1, alternate_signal=False)
        # 인덱스 2: 0+1=1 초과 → 무효
        assert result.iloc[2] == 0


# ---------------------------------------------------------------------------
# Alternate Signal 테스트
# ---------------------------------------------------------------------------


class TestAlternateSignal:
    """Alternate Signal 필터링 검증."""

    def test_consecutive_long_filtered(self, gen: SignalGenerator):
        """연속 Long → 두 번째 Long은 0으로 필터링."""
        leading = _make_result(
            [True, False, True, False],
            [False, False, False, False],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=True)
        assert result.iloc[0] == 1
        assert result.iloc[2] == 0  # 연속 Long 필터링

    def test_consecutive_short_filtered(self, gen: SignalGenerator):
        """연속 Short → 두 번째 Short은 0으로 필터링."""
        leading = _make_result(
            [False, False, False, False],
            [True, False, True, False],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=True)
        assert result.iloc[0] == -1
        assert result.iloc[2] == 0  # 연속 Short 필터링

    def test_alternating_signals_preserved(self, gen: SignalGenerator):
        """Long → Short → Long 교대 시그널은 모두 유지."""
        leading = _make_result(
            [True, False, False, True, False],
            [False, False, True, False, False],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=True)
        assert result.iloc[0] == 1   # Long
        assert result.iloc[2] == -1  # Short (방향 전환)
        assert result.iloc[3] == 1   # Long (방향 전환)

    def test_alternate_signal_disabled(self, gen: SignalGenerator):
        """alternate_signal=False이면 연속 동일 방향 허용."""
        leading = _make_result(
            [True, False, True, False],
            [False, False, False, False],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=False)
        assert result.iloc[0] == 1
        assert result.iloc[2] == 1  # 필터링 안 됨


# ---------------------------------------------------------------------------
# 엣지 케이스 테스트
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """엣지 케이스 검증."""

    def test_empty_series(self, gen: SignalGenerator):
        """빈 시리즈 입력 시 빈 시리즈 반환."""
        idx = pd.DatetimeIndex([])
        leading = IndicatorResult(
            long_signal=pd.Series(dtype=bool, index=idx),
            short_signal=pd.Series(dtype=bool, index=idx),
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=True)
        assert len(result) == 0

    def test_all_false_signals(self, gen: SignalGenerator):
        """모든 시그널이 False이면 전체 0."""
        leading = _make_result(
            [False, False, False, False],
            [False, False, False, False],
        )
        conf = _make_result(
            [False, False, False, False],
            [False, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=3, alternate_signal=True)
        assert (result == 0).all()

    def test_return_values_in_valid_set(self, gen: SignalGenerator):
        """반환값이 항상 {-1, 0, 1} 중 하나."""
        leading = _make_result(
            [True, False, True, False, True],
            [False, True, False, True, False],
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=True)
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_single_element_series(self, gen: SignalGenerator):
        """길이 1 시리즈도 정상 처리."""
        leading = _make_result([True], [False])
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=True)
        assert len(result) == 1
        assert result.iloc[0] == 1

    def test_nan_in_signals_treated_as_false(self, gen: SignalGenerator):
        """NaN 값은 False로 처리."""
        idx = _make_index(3)
        leading = IndicatorResult(
            long_signal=pd.Series([True, np.nan, False], index=idx),
            short_signal=pd.Series([False, False, np.nan], index=idx),
        )
        result = gen.generate(leading, [], signal_expiry=3, alternate_signal=False)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 0  # NaN → False → 0
        assert result.iloc[2] == 0  # NaN → False → 0

    def test_immediate_confirmation_on_leading_fire(self, gen: SignalGenerator):
        """리딩 시그널과 확인 시그널이 동시에 발생하면 즉시 유효."""
        leading = _make_result(
            [True, False, False],
            [False, False, False],
        )
        conf = _make_result(
            [True, False, False],
            [False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=3, alternate_signal=False)
        assert result.iloc[0] == 1

    def test_multiple_leading_fires_reset_expiry(self, gen: SignalGenerator):
        """새로운 리딩 시그널이 발생하면 expiry 윈도우가 리셋된다."""
        # 리딩: 인덱스 0에서 Long, 인덱스 3에서 다시 Long
        # 확인: 인덱스 5에서 Long
        # 인덱스 0 기준 expiry=3 → 인덱스 3까지 유효 → 인덱스 5는 무효
        # 인덱스 3 기준 expiry=3 → 인덱스 6까지 유효 → 인덱스 5는 유효
        leading = _make_result(
            [True, False, False, True, False, False, False],
            [False, False, False, False, False, False, False],
        )
        conf = _make_result(
            [False, False, False, False, False, True, False],
            [False, False, False, False, False, False, False],
        )
        result = gen.generate(leading, [conf], signal_expiry=3, alternate_signal=False)
        assert result.iloc[5] == 1  # 인덱스 3 기준 expiry 내
