"""조합 생성 엔진 단위 테스트.

CombinationEngine의 조합 생성, 고유 ID, 자동 축소 로직,
필터링 규칙(제외 지표, 리딩-확인 겹침, 분류 중복, 파라미터 프리셋)을 검증한다.
"""

from math import comb

import pytest

from sp500_backtest.engine.combination import (
    CONFIRMATION_CATEGORIES,
    EXCLUDED_INDICATORS,
    LEADING_PARAM_PRESETS,
    CombinationEngine,
    IndicatorCombination,
    _ExpandedConfirmation,
    _conf_base_name,
    _format_param_suffix,
    _get_conf_base_for_leading,
)


# ---------------------------------------------------------------------------
# 테스트 픽스처
# ---------------------------------------------------------------------------

# 레거시 모드 옵션: 필터링/프리셋 비활성화
_LEGACY_OPTS = {"excluded_names": set(), "category_map": {}, "param_presets": {}}


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
    """소규모 데이터로 초기화된 CombinationEngine (레거시 모드)."""
    return CombinationEngine(small_leading, small_confirmations, **_LEGACY_OPTS)


@pytest.fixture
def no_subtype_confirmations() -> list[dict]:
    """서브타입이 없는 확인 지표 정보 (2개)."""
    return [
        {"name": "EMA Filter Confirmation", "subtypes": []},
        {"name": "ROC Confirmation", "subtypes": []},
    ]


# ---------------------------------------------------------------------------
# 유틸리티 함수 테스트
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """유틸리티 함수 검증."""

    def test_conf_base_name_with_suffix(self):
        """' Confirmation' 접미사가 있으면 제거한다."""
        assert _conf_base_name("TSI Confirmation") == "TSI"
        assert _conf_base_name("Range Filter Confirmation") == "Range Filter"

    def test_conf_base_name_without_suffix(self):
        """접미사가 없으면 원본 그대로 반환한다."""
        assert _conf_base_name("EMA Filter") == "EMA Filter"

    def test_get_conf_base_for_leading_mapped(self):
        """매핑된 리딩 지표는 대응하는 확인 base name을 반환한다."""
        assert _get_conf_base_for_leading("DMI") == "DMI ADX"
        assert _get_conf_base_for_leading("Vortex Index") == "Vortex Indicator"

    def test_get_conf_base_for_leading_default(self):
        """매핑되지 않은 리딩 지표는 이름 그대로 반환한다."""
        assert _get_conf_base_for_leading("Range Filter") == "Range Filter"
        assert _get_conf_base_for_leading("MACD") == "MACD"

    def test_format_param_suffix_empty(self):
        """빈 파라미터는 빈 문자열을 반환한다."""
        assert _format_param_suffix({}) == ""

    def test_format_param_suffix_with_params(self):
        """파라미터가 있으면 축약 접미사를 반환한다."""
        result = _format_param_suffix({"period": 50, "mult": 2.0})
        assert result.startswith("_")
        assert "p50" in result
        assert "m2" in result


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
        engine = CombinationEngine(["A"], no_subtype_confirmations, **_LEGACY_OPTS)
        assert len(engine._expanded) == 2
        assert all(ec.subtype is None for ec in engine._expanded)

    def test_expand_with_subtypes(self, small_confirmations: list[dict]):
        """서브타입이 있는 지표는 서브타입 수만큼 확장된다."""
        engine = CombinationEngine(["A"], small_confirmations, **_LEGACY_OPTS)
        # EMA Filter(1) + TSI(2) + RSI(2) = 5
        assert len(engine._expanded) == 5

    def test_expanded_labels_unique(self, small_confirmations: list[dict]):
        """확장된 항목의 short_label이 모두 고유하다."""
        engine = CombinationEngine(["A"], small_confirmations, **_LEGACY_OPTS)
        labels = [ec.short_label for ec in engine._expanded]
        assert len(labels) == len(set(labels))

    def test_expanded_has_base_name_and_category(self):
        """확장된 항목에 base_name과 category가 설정된다."""
        confs = [{"name": "TSI Confirmation", "subtypes": ["Signal Cross"]}]
        cat_map = {"TSI": "모멘텀"}
        engine = CombinationEngine(["A"], confs, excluded_names=set(), category_map=cat_map, param_presets={})
        assert engine._expanded[0].base_name == "TSI"
        assert engine._expanded[0].category == "모멘텀"


# ---------------------------------------------------------------------------
# _estimate_total 테스트 (레거시 모드: 프리셋 없음)
# ---------------------------------------------------------------------------


class TestEstimateTotal:
    """조합 수 추정 로직 검증 (레거시 모드)."""

    def test_zero_confirmations(self, engine: CombinationEngine):
        """max_confirmations=0이면 리딩 지표 수만큼 조합이 생성된다."""
        # L=3, P=1(프리셋 없음), C(5,0)=1 → 3×1×1=3
        assert engine._estimate_total(0) == 3

    def test_one_confirmation(self, engine: CombinationEngine):
        """max_confirmations=1이면 L × (C(C,0) + C(C,1)) 조합이 생성된다."""
        # L=3, P=1, C=5: C(5,0)+C(5,1) = 1+5 = 6 → 3×1×6=18
        assert engine._estimate_total(1) == 18

    def test_two_confirmations(self, engine: CombinationEngine):
        """max_confirmations=2이면 L × (C(C,0)+C(C,1)+C(C,2)) 조합이 생성된다."""
        # L=3, P=1, C=5: C(5,0)+C(5,1)+C(5,2) = 1+5+10 = 16 → 3×1×16=48
        assert engine._estimate_total(2) == 48

    def test_formula_matches_manual(self, engine: CombinationEngine):
        """공식 L × P × Σ(k=0..N) C(C,k)와 수동 계산이 일치한다."""
        l_count = 3
        c_count = 5
        for n in range(c_count + 1):
            expected = l_count * sum(comb(c_count, k) for k in range(n + 1))
            assert engine._estimate_total(n) == expected

    def test_max_confirmations_exceeds_count(self, engine: CombinationEngine):
        """max_confirmations가 확인 지표 수보다 크면 전체 조합(2^C)이 된다."""
        # L=3, P=1, C=5: Σ(k=0..5) C(5,k) = 2^5 = 32 → 3×1×32=96
        assert engine._estimate_total(10) == 96
        assert engine._estimate_total(100) == 96


# ---------------------------------------------------------------------------
# generate 테스트 (레거시 모드)
# ---------------------------------------------------------------------------


class TestGenerate:
    """조합 생성 로직 검증 (레거시 모드)."""

    def test_returns_list_of_combinations(self, engine: CombinationEngine):
        """generate()가 IndicatorCombination 리스트를 반환한다."""
        result = engine.generate(max_confirmations=1)
        assert isinstance(result, list)
        assert all(isinstance(c, IndicatorCombination) for c in result)

    def test_count_matches_estimate(self, engine: CombinationEngine):
        """생성된 조합 수가 _estimate_total과 일치한다 (레거시 모드)."""
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
        assert len(leading_only) == 3  # 리딩 지표 3개 × 프리셋 1개

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
        assert len(subtype_combos) > 0

    def test_id_format_leading_only(self, engine: CombinationEngine):
        """확인 지표 없는 조합의 ID는 '_conf-' 구분자를 포함하지 않는다."""
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
        engine = CombinationEngine(leading, confs, **_LEGACY_OPTS)
        result = engine.generate(max_confirmations=3, max_combinations=100)
        assert len(result) <= 100

    def test_auto_reduce_to_zero(self):
        """max_combinations가 매우 작으면 max_confirmations=0까지 축소된다."""
        leading = [f"Leading{i}" for i in range(10)]
        confs = [{"name": f"Conf{i}", "subtypes": []} for i in range(5)]
        engine = CombinationEngine(leading, confs, **_LEGACY_OPTS)
        result = engine.generate(max_confirmations=3, max_combinations=10)
        assert len(result) == 10
        assert all(len(c.confirmations) == 0 for c in result)

    def test_empty_confirmations(self):
        """확인 지표가 없으면 리딩 지표만으로 조합이 생성된다."""
        engine = CombinationEngine(["A", "B"], [], **_LEGACY_OPTS)
        result = engine.generate(max_confirmations=3)
        assert len(result) == 2
        assert all(len(c.confirmations) == 0 for c in result)

    def test_single_leading_single_confirmation(self):
        """리딩 1개, 확인 1개(서브타입 없음)로 정확한 조합이 생성된다."""
        engine = CombinationEngine(
            ["RSI"],
            [{"name": "EMA Filter Confirmation", "subtypes": []}],
            **_LEGACY_OPTS,
        )
        # L=1, P=1, C=1: C(1,0)+C(1,1) = 2 → 1×1×2=2
        result = engine.generate(max_confirmations=1)
        assert len(result) == 2

    def test_console_output(self, engine: CombinationEngine, capsys: pytest.CaptureFixture):
        """generate()가 총 조합 수를 콘솔에 출력한다."""
        engine.generate(max_confirmations=1)
        captured = capsys.readouterr()
        assert "총 조합 수:" in captured.out


# ---------------------------------------------------------------------------
# 필터링 규칙 테스트
# ---------------------------------------------------------------------------


class TestExcludedIndicators:
    """제외 지표 필터링 검증."""

    def test_excluded_leading_removed(self):
        """제외 대상 리딩 지표가 조합에서 제거된다."""
        leading = ["Range Filter", "Wolfpack Id", "HACOLT", "SuperIchi", "B-Xtrender"]
        engine = CombinationEngine(leading, [], excluded_names=EXCLUDED_INDICATORS, category_map={}, param_presets={})
        result = engine.generate(max_confirmations=0)
        leading_names = {c.leading for c in result}
        assert "Wolfpack Id" not in leading_names
        assert "HACOLT" not in leading_names
        assert "SuperIchi" not in leading_names
        assert "B-Xtrender" not in leading_names
        assert "Range Filter" in leading_names

    def test_excluded_confirmation_removed(self):
        """제외 대상 확인 지표가 조합에서 제거된다."""
        confs = [
            {"name": "EMA Filter Confirmation", "subtypes": []},
            {"name": "Wolfpack Id Confirmation", "subtypes": []},
            {"name": "HACOLT Confirmation", "subtypes": []},
        ]
        engine = CombinationEngine(["A"], confs, excluded_names=EXCLUDED_INDICATORS, category_map={}, param_presets={})
        assert len(engine._expanded) == 1  # EMA Filter만 남음
        assert engine._expanded[0].base_name == "EMA Filter"


class TestLeadingConfirmationOverlap:
    """리딩-확인 지표 겹침 방지 검증."""

    def test_same_indicator_excluded(self):
        """리딩 지표와 동일 계열 확인 지표가 같은 조합에 포함되지 않는다."""
        confs = [
            {"name": "Range Filter Confirmation", "subtypes": []},
            {"name": "TSI Confirmation", "subtypes": []},
        ]
        engine = CombinationEngine(
            ["Range Filter"],
            confs,
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1)
        for combo in result:
            for conf in combo.confirmations:
                # Range Filter 리딩 + Range Filter 확인 조합은 없어야 함
                assert _conf_base_name(conf["name"]) != "Range Filter"

    def test_dmi_vortex_mapping(self):
        """DMI 리딩과 DMI ADX 확인이 겹침으로 처리된다."""
        confs = [
            {"name": "DMI ADX Confirmation", "subtypes": []},
            {"name": "TSI Confirmation", "subtypes": []},
        ]
        engine = CombinationEngine(
            ["DMI"],
            confs,
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1)
        for combo in result:
            for conf in combo.confirmations:
                assert _conf_base_name(conf["name"]) != "DMI ADX"

    def test_non_overlapping_allowed(self):
        """리딩과 다른 계열 확인 지표는 같은 조합에 포함된다."""
        confs = [{"name": "TSI Confirmation", "subtypes": []}]
        engine = CombinationEngine(
            ["Range Filter"],
            confs,
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1)
        with_conf = [c for c in result if len(c.confirmations) > 0]
        assert len(with_conf) == 1  # Range Filter + TSI 조합 존재


class TestCategoryFilter:
    """실무형 분류 중복 방지 검증."""

    def test_same_category_two_excluded(self):
        """동일 분류에서 2개 이상 확인 지표가 포함되지 않는다."""
        confs = [
            {"name": "RSI Confirmation", "subtypes": []},
            {"name": "MACD Confirmation", "subtypes": []},
            {"name": "TSI Confirmation", "subtypes": []},
        ]
        cat_map = {"RSI": "모멘텀", "MACD": "모멘텀", "TSI": "모멘텀"}
        engine = CombinationEngine(
            ["Range Filter"],
            confs,
            excluded_names=set(),
            category_map=cat_map,
            param_presets={},
        )
        result = engine.generate(max_confirmations=3)
        for combo in result:
            if len(combo.confirmations) > 1:
                categories = [
                    cat_map.get(_conf_base_name(c["name"]))
                    for c in combo.confirmations
                ]
                # None이 아닌 카테고리만 필터링
                non_none = [c for c in categories if c is not None]
                # 동일 카테고리가 2개 이상이면 안 됨
                assert len(non_none) == len(set(non_none)), (
                    f"동일 분류 중복: {combo.id}"
                )

    def test_different_categories_allowed(self):
        """서로 다른 분류의 확인 지표는 같은 조합에 포함된다."""
        confs = [
            {"name": "RSI Confirmation", "subtypes": []},
            {"name": "Supertrend Confirmation", "subtypes": []},
            {"name": "Volume Confirmation", "subtypes": []},
        ]
        cat_map = {"RSI": "모멘텀", "Supertrend": "추세", "Volume": "거래량"}
        engine = CombinationEngine(
            ["Range Filter"],
            confs,
            excluded_names=set(),
            category_map=cat_map,
            param_presets={},
        )
        result = engine.generate(max_confirmations=3)
        # 3개 모두 다른 분류이므로 3개 조합이 가능해야 함
        three_conf = [c for c in result if len(c.confirmations) == 3]
        assert len(three_conf) == 1

    def test_leading_category_overlap_allowed(self):
        """리딩 지표와 확인 지표의 분류가 같아도 허용된다 (분류 제한은 확인 지표 간에만)."""
        confs = [
            {"name": "Supertrend Confirmation", "subtypes": []},
        ]
        cat_map = {"Supertrend": "추세"}
        # Range Filter도 추세 분류이지만, 리딩-확인 분류 겹침은 허용
        engine = CombinationEngine(
            ["Range Filter"],
            confs,
            excluded_names=set(),
            category_map=cat_map,
            param_presets={},
        )
        result = engine.generate(max_confirmations=1)
        with_conf = [c for c in result if len(c.confirmations) > 0]
        assert len(with_conf) == 1


class TestParamPresets:
    """파라미터 프리셋 조합 생성 검증."""

    def test_presets_multiply_combinations(self):
        """파라미터 프리셋이 조합 수를 배수로 증가시킨다."""
        presets = {
            "RSI": [
                {},
                {"length": 9},
                {"length": 21},
            ]
        }
        engine = CombinationEngine(
            ["RSI"],
            [{"name": "EMA Filter Confirmation", "subtypes": []}],
            excluded_names=set(),
            category_map={},
            param_presets=presets,
        )
        result = engine.generate(max_confirmations=1)
        # 프리셋 3개 × (확인 0개 + 확인 1개) = 3 × 2 = 6
        assert len(result) == 6

    def test_preset_params_in_result(self):
        """생성된 조합에 파라미터 프리셋 값이 포함된다."""
        presets = {"RSI": [{}, {"length": 9}]}
        engine = CombinationEngine(
            ["RSI"],
            [],
            excluded_names=set(),
            category_map={},
            param_presets=presets,
        )
        result = engine.generate(max_confirmations=0)
        assert len(result) == 2
        params_list = [c.leading_params for c in result]
        assert {} in params_list
        assert {"length": 9} in params_list

    def test_preset_ids_unique(self):
        """파라미터 프리셋이 다른 조합은 서로 다른 ID를 가진다."""
        presets = {"RSI": [{}, {"length": 9}, {"length": 21}]}
        engine = CombinationEngine(
            ["RSI"],
            [],
            excluded_names=set(),
            category_map={},
            param_presets=presets,
        )
        result = engine.generate(max_confirmations=0)
        ids = [c.id for c in result]
        assert len(ids) == len(set(ids))

    def test_no_preset_uses_default(self):
        """프리셋이 정의되지 않은 지표는 기본 파라미터({})로 1개만 생성된다."""
        engine = CombinationEngine(
            ["CustomIndicator"],
            [],
            excluded_names=set(),
            category_map={},
            param_presets={},  # 프리셋 없음
        )
        result = engine.generate(max_confirmations=0)
        assert len(result) == 1
        assert result[0].leading_params == {}


class TestDefaultConstants:
    """기본 상수 정의 검증."""

    def test_excluded_indicators_contains_required(self):
        """EXCLUDED_INDICATORS에 필수 제외 지표가 포함되어 있다."""
        assert "Wolfpack Id" in EXCLUDED_INDICATORS
        assert "HACOLT" in EXCLUDED_INDICATORS
        assert "SuperIchi" in EXCLUDED_INDICATORS
        assert "B-Xtrender" in EXCLUDED_INDICATORS

    def test_categories_cover_all_types(self):
        """CONFIRMATION_CATEGORIES에 4개 분류가 모두 존재한다."""
        categories = set(CONFIRMATION_CATEGORIES.values())
        assert "추세" in categories
        assert "모멘텀" in categories
        assert "추세강도" in categories
        assert "거래량" in categories

    def test_param_presets_have_default(self):
        """모든 파라미터 프리셋에 기본값({})이 포함되어 있다."""
        for name, presets in LEADING_PARAM_PRESETS.items():
            assert {} in presets, f"{name}의 프리셋에 기본값({{}})이 없음"

    def test_param_presets_have_variations(self):
        """모든 파라미터 프리셋에 2개 이상의 변형이 있다."""
        for name, presets in LEADING_PARAM_PRESETS.items():
            assert len(presets) >= 2, f"{name}의 프리셋이 2개 미만"


class TestSignalExpiryValues:
    """signal_expiry_values 파라미터에 의한 조합 복제 검증."""

    def test_no_expiry_values_default_behavior(self):
        """signal_expiry_values가 None이면 기존 동작과 동일하다."""
        engine = CombinationEngine(
            ["A"],
            [{"name": "X Confirmation", "subtypes": []}],
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1, signal_expiry_values=None)
        # A만, 확인 X 1개 → leading_only(1) + with_conf(1) = 2
        assert len(result) == 2
        for combo in result:
            assert combo.signal_expiry is None

    def test_single_expiry_value_no_duplication(self):
        """signal_expiry_values가 1개면 복제하지 않는다."""
        engine = CombinationEngine(
            ["A"],
            [{"name": "X Confirmation", "subtypes": []}],
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1, signal_expiry_values=[3])
        assert len(result) == 2
        for combo in result:
            assert combo.signal_expiry is None

    def test_multiple_expiry_values_multiply_conf_combos(self):
        """signal_expiry_values가 여러 개면 확인 지표 있는 조합만 복제된다."""
        engine = CombinationEngine(
            ["A"],
            [{"name": "X Confirmation", "subtypes": []}],
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1, signal_expiry_values=[1, 3, 5])
        # leading_only(1, 복제 안 됨) + with_conf(1) × 3 expiry = 4
        assert len(result) == 4

    def test_expiry_value_in_combo_id(self):
        """signal_expiry 값이 조합 ID에 |se=N 형태로 포함된다."""
        engine = CombinationEngine(
            ["A"],
            [{"name": "X Confirmation", "subtypes": []}],
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1, signal_expiry_values=[2, 5])
        conf_combos = [c for c in result if c.confirmations]
        assert len(conf_combos) == 2
        ids = {c.id for c in conf_combos}
        assert any("|se=2" in cid for cid in ids)
        assert any("|se=5" in cid for cid in ids)

    def test_expiry_value_stored_in_combination(self):
        """각 복제된 조합에 올바른 signal_expiry 값이 저장된다."""
        engine = CombinationEngine(
            ["A"],
            [{"name": "X Confirmation", "subtypes": []}],
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1, signal_expiry_values=[1, 3, 7])
        conf_combos = [c for c in result if c.confirmations]
        expiry_values = {c.signal_expiry for c in conf_combos}
        assert expiry_values == {1, 3, 7}

    def test_leading_only_not_duplicated(self):
        """확인 지표 없는 조합은 signal_expiry_values와 무관하게 1개만 생성된다."""
        engine = CombinationEngine(
            ["A", "B"],
            [{"name": "X Confirmation", "subtypes": []}],
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=1, signal_expiry_values=[1, 3, 5, 7])
        leading_only = [c for c in result if not c.confirmations]
        # A, B 각각 1개씩 = 2개 (복제 안 됨)
        assert len(leading_only) == 2
        for combo in leading_only:
            assert combo.signal_expiry is None

    def test_all_ids_unique_with_expiry(self):
        """signal_expiry 변형 포함 시 모든 조합 ID가 고유하다."""
        engine = CombinationEngine(
            ["A", "B"],
            [
                {"name": "X Confirmation", "subtypes": []},
                {"name": "Y Confirmation", "subtypes": []},
            ],
            excluded_names=set(),
            category_map={},
            param_presets={},
        )
        result = engine.generate(max_confirmations=2, signal_expiry_values=[1, 3, 5])
        ids = [c.id for c in result]
        assert len(ids) == len(set(ids))
