"""조합 생성 엔진 단위 테스트.

CombinationEngine의 조합 생성, 고유 ID, 자동 축소 로직을 검증한다.
"""

from math import comb

import pytest

from sp500_backtest.engine.combination import (
    CombinationEngine,
    IndicatorCombination,
    _ExpandedConfirmation,
)


# ---------------------------------------------------------------------------
# 테스트 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture
def small_leading() -> list[str]:
    """소규모 리딩 지표 이름 목록 (3개)."""
    return ["Range Filter", "Supertrend", "MACD"]


@pytest.fixture
def small_confirmations() -> list[dict]:
    """소규모 확인 지표 정보 (서브타입 포함, 3개 지표 → 확장 시 5개)."""
    return [
        {"name": "EMA Filter Confirmation", "subtypes": []},
        {"name": "TSI Confirmation", "subtypes": ["Signal Cross", "Zero line cross"]},
        {"name": "RSI Confirmation", "subtypes": ["RSI MA Cross", "RSI Level"]},
    ]


@pytest.fixture
def engine(small_leading: list[str], small_confirmations: list[dict]) -> CombinationEngine:
    """소규모 데이터로 초기화된 CombinationEngine 인스턴스."""
    return CombinationEngine(small_leading, small_confirmations)


@pytest.fixture
def no_subtype_confirmations() -> list[dict]:
    """서브타입이 없는 확인 지표 정보 (2개)."""
    return [
        {"name": "EMA Filter Confirmation", "subtypes": []},
        {"name": "ROC Confirmation", "subtypes": []},
    ]


# ---------------------------------------------------------------------------
# IndicatorCombination 테스트
# ---------------------------------------------------------------------------


class TestIndicatorCombination:
    """IndicatorCombination dataclass 검증."""

    def test_create_with_defaults(self):
        """기본값으로 정상 생성된다."""
        combo = IndicatorCombination(id="test", leading="RSI")
        assert combo.id == "test"
        assert combo.leading == "RSI"
        assert combo.leading_params == {}
        assert combo.confirmations == []

    def test_create_with_confirmations(self):
        """확인 지표 리스트를 포함하여 생성된다."""
        confs = [{"name": "EMA Filter Confirmation", "params": {}}]
        combo = IndicatorCombination(
            id="RSI_conf-EMAFilter",
            leading="RSI",
            leading_params={"length": 14},
            confirmations=confs,
        )
        assert len(combo.confirmations) == 1
        assert combo.leading_params == {"length": 14}


# ---------------------------------------------------------------------------
# _ExpandedConfirmation 테스트
# ---------------------------------------------------------------------------


class TestExpandedConfirmation:
    """서브타입 확장 로직 검증."""

    def test_expand_no_subtypes(self, no_subtype_confirmations: list[dict]):
        """서브타입이 없는 지표는 1개 항목으로 확장된다."""
        engine = CombinationEngine(["A"], no_subtype_confirmations)
        assert len(engine._expanded) == 2
        assert all(ec.subtype is None for ec in engine._expanded)

    def test_expand_with_subtypes(self, small_confirmations: list[dict]):
        """서브타입이 있는 지표는 서브타입 수만큼 확장된다."""
        engine = CombinationEngine(["A"], small_confirmations)
        # EMA Filter(1) + TSI(2) + RSI(2) = 5
        assert len(engine._expanded) == 5

    def test_expanded_labels_unique(self, small_confirmations: list[dict]):
        """확장된 항목의 short_label이 모두 고유하다."""
        engine = CombinationEngine(["A"], small_confirmations)
        labels = [ec.short_label for ec in engine._expanded]
        assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# _estimate_total 테스트
# ---------------------------------------------------------------------------


class TestEstimateTotal:
    """조합 수 추정 로직 검증."""

    def test_zero_confirmations(self, engine: CombinationEngine):
        """max_confirmations=0이면 리딩 지표 수만큼 조합이 생성된다."""
        # L=3, C(5,0)=1 → 3×1=3
        assert engine._estimate_total(0) == 3

    def test_one_confirmation(self, engine: CombinationEngine):
        """max_confirmations=1이면 L × (C(C,0) + C(C,1)) 조합이 생성된다."""
        # L=3, C=5: C(5,0)+C(5,1) = 1+5 = 6 → 3×6=18
        assert engine._estimate_total(1) == 18

    def test_two_confirmations(self, engine: CombinationEngine):
        """max_confirmations=2이면 L × (C(C,0)+C(C,1)+C(C,2)) 조합이 생성된다."""
        # L=3, C=5: C(5,0)+C(5,1)+C(5,2) = 1+5+10 = 16 → 3×16=48
        assert engine._estimate_total(2) == 48

    def test_formula_matches_manual(self, engine: CombinationEngine):
        """공식 L × Σ(k=0..N) C(C,k)와 수동 계산이 일치한다."""
        l_count = 3
        c_count = 5
        for n in range(c_count + 1):
            expected = l_count * sum(comb(c_count, k) for k in range(n + 1))
            assert engine._estimate_total(n) == expected

    def test_max_confirmations_exceeds_count(self, engine: CombinationEngine):
        """max_confirmations가 확인 지표 수보다 크면 전체 조합(2^C)이 된다."""
        # L=3, C=5: Σ(k=0..5) C(5,k) = 2^5 = 32 → 3×32=96
        assert engine._estimate_total(10) == 96
        assert engine._estimate_total(100) == 96


# ---------------------------------------------------------------------------
# generate 테스트
# ---------------------------------------------------------------------------


class TestGenerate:
    """조합 생성 로직 검증."""

    def test_returns_list_of_combinations(self, engine: CombinationEngine):
        """generate()가 IndicatorCombination 리스트를 반환한다."""
        result = engine.generate(max_confirmations=1)
        assert isinstance(result, list)
        assert all(isinstance(c, IndicatorCombination) for c in result)

    def test_count_matches_estimate(self, engine: CombinationEngine):
        """생성된 조합 수가 _estimate_total과 일치한다."""
        for n in range(4):
            result = engine.generate(max_confirmations=n, max_combinations=1_000_000)
            expected = engine._estimate_total(n)
            assert len(result) == expected, f"max_confirmations={n}: {len(result)} != {expected}"

    def test_all_ids_unique(self, engine: CombinationEngine):
        """모든 조합의 ID가 고유하다."""
        result = engine.generate(max_confirmations=2)
        ids = [c.id for c in result]
        assert len(ids) == len(set(ids)), f"중복 ID 발견: {len(ids)} != {len(set(ids))}"

    def test_leading_only_combinations(self, engine: CombinationEngine):
        """확인 지표 0개 조합이 리딩 지표 수만큼 존재한다."""
        result = engine.generate(max_confirmations=2)
        leading_only = [c for c in result if len(c.confirmations) == 0]
        assert len(leading_only) == 3  # 리딩 지표 3개

    def test_confirmation_entries_have_name(self, engine: CombinationEngine):
        """모든 확인 지표 항목에 name 키가 존재한다."""
        result = engine.generate(max_confirmations=1)
        for combo in result:
            for conf in combo.confirmations:
                assert "name" in conf
                assert "params" in conf

    def test_subtype_in_confirmation_entry(self, engine: CombinationEngine):
        """서브타입이 있는 확인 지표 항목에 subtype 키가 존재한다."""
        result = engine.generate(max_confirmations=1)
        subtype_combos = [
            c for c in result
            if any("subtype" in conf for conf in c.confirmations)
        ]
        assert len(subtype_combos) > 0  # 서브타입 조합이 존재해야 함

    def test_id_format_leading_only(self, engine: CombinationEngine):
        """확인 지표 없는 조합의 ID는 리딩 지표 라벨만 포함한다."""
        result = engine.generate(max_confirmations=1)
        leading_only = [c for c in result if len(c.confirmations) == 0]
        for combo in leading_only:
            assert "_conf-" not in combo.id

    def test_id_format_with_confirmations(self, engine: CombinationEngine):
        """확인 지표가 있는 조합의 ID는 '_conf-' 구분자를 포함한다."""
        result = engine.generate(max_confirmations=1)
        with_conf = [c for c in result if len(c.confirmations) > 0]
        for combo in with_conf:
            assert "_conf-" in combo.id

    def test_auto_reduce_max_confirmations(self):
        """조합 수가 max_combinations를 초과하면 max_confirmations가 자동 축소된다."""
        leading = [f"Leading{i}" for i in range(5)]
        confs = [{"name": f"Conf{i}", "subtypes": []} for i in range(10)]
        engine = CombinationEngine(leading, confs)

        # max_confirmations=3이면 5 × (C(10,0)+C(10,1)+C(10,2)+C(10,3)) = 5×176 = 880
        # max_confirmations=10이면 5 × 2^10 = 5120
        # max_combinations=100으로 제한하면 축소 발생
        result = engine.generate(max_confirmations=3, max_combinations=100)
        assert len(result) <= 100

    def test_auto_reduce_to_zero(self):
        """max_combinations가 매우 작으면 max_confirmations=0까지 축소된다."""
        leading = [f"Leading{i}" for i in range(10)]
        confs = [{"name": f"Conf{i}", "subtypes": []} for i in range(5)]
        engine = CombinationEngine(leading, confs)

        # max_confirmations=0이면 10개 조합
        # max_combinations=10이면 0까지 축소
        result = engine.generate(max_confirmations=3, max_combinations=10)
        assert len(result) == 10
        assert all(len(c.confirmations) == 0 for c in result)

    def test_empty_confirmations(self):
        """확인 지표가 없으면 리딩 지표만으로 조합이 생성된다."""
        engine = CombinationEngine(["A", "B"], [])
        result = engine.generate(max_confirmations=3)
        assert len(result) == 2
        assert all(len(c.confirmations) == 0 for c in result)

    def test_single_leading_single_confirmation(self):
        """리딩 1개, 확인 1개(서브타입 없음)로 정확한 조합이 생성된다."""
        engine = CombinationEngine(
            ["RSI"],
            [{"name": "EMA Filter Confirmation", "subtypes": []}],
        )
        # L=1, C=1: C(1,0)+C(1,1) = 2 → 1×2=2
        result = engine.generate(max_confirmations=1)
        assert len(result) == 2

    def test_console_output(self, engine: CombinationEngine, capsys: pytest.CaptureFixture):
        """generate()가 총 조합 수를 콘솔에 출력한다."""
        engine.generate(max_confirmations=1)
        captured = capsys.readouterr()
        assert "총 조합 수:" in captured.out
