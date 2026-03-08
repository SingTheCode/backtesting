"""지표 조합 생성 엔진 모듈.

리딩 지표와 확인 지표(서브타입 포함)의 모든 가능한 조합을
생성하고 관리한다. 조합 수가 상한을 초과하면 max_confirmations를
자동으로 축소한다.

필터링 규칙:
- 제외 지표(EXCLUDED_INDICATORS)는 리딩/확인 모두에서 제거
- 리딩 지표와 동일 계열 확인 지표는 같은 조합에 포함 불가
- 확인 지표 중 동일 실무형 분류(CONFIRMATION_CATEGORIES)에서 2개 이상 불가
- 리딩 지표 파라미터 프리셋(LEADING_PARAM_PRESETS)으로 파라미터 변형 조합 생성
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

# ---------------------------------------------------------------------------
# 제외 지표 목록 (리딩 + 확인 모두 제거)
# ---------------------------------------------------------------------------
EXCLUDED_INDICATORS: set[str] = {
    "Wolfpack Id",
    "HACOLT",
    "SuperIchi",
    "B-Xtrender",
}

# ---------------------------------------------------------------------------
# 확인 지표 실무형 분류 매핑 (확인 지표 base name → 카테고리)
# base name = 확인 지표 이름에서 " Confirmation" 접미사를 제거한 이름
# ---------------------------------------------------------------------------
CONFIRMATION_CATEGORIES: dict[str, str] = {
    # 추세
    "EMA Filter": "추세",
    "2 EMA Cross": "추세",
    "3 EMA Cross": "추세",
    "Range Filter": "추세",
    "RQK": "추세",
    "Supertrend": "추세",
    "Half Trend": "추세",
    "Donchian Trend Ribbon": "추세",
    "McGinley Dynamic": "추세",
    "Ichimoku Cloud": "추세",
    "Trendline Breakout": "추세",
    "Range Detector": "추세",
    "Chandelier Exit": "추세",
    "Parabolic SAR": "추세",
    "SSL Channel": "추세",
    "Hull Suite": "추세",
    # 모멘텀
    "ROC": "모멘텀",
    "DPO": "모멘텀",
    "TSI": "모멘텀",
    "BB Oscillator": "모멘텀",
    "Stochastic": "모멘텀",
    "RSI": "모멘텀",
    "RSI MA Direction": "모멘텀",
    "RSI Limit": "모멘텀",
    "RSI MA Limit": "모멘텀",
    "MACD": "모멘텀",
    "CCI": "모멘텀",
    "Awesome Oscillator": "모멘텀",
    "Waddah Attar Explosion": "모멘텀",
    "QQE Mod": "모멘텀",
    "STC": "모멘텀",
    # 추세강도
    "Bull Bear Power Trend": "추세강도",
    "DMI ADX": "추세강도",  # 확인 지표 이름: "DMI ADX Confirmation"
    "Vortex Indicator": "추세강도",  # 확인 지표 이름: "Vortex Indicator Confirmation"
    "Choppiness Index": "추세강도",
    "Damiani Volatility": "추세강도",
    "Volatility Oscillator": "추세강도",
    # 거래량/세션
    "Volume": "거래량",
    "Chaikin Money Flow": "거래량",
    "VWAP": "거래량",
}

# ---------------------------------------------------------------------------
# 리딩 지표 → 확인 지표 base name 매핑 (이름이 다른 경우 명시)
# 대부분은 리딩 이름 == 확인 base name 이지만, 일부 불일치 존재
# ---------------------------------------------------------------------------
LEADING_TO_CONF_BASE: dict[str, str] = {
    "DMI": "DMI ADX",  # 리딩 "DMI" ↔ 확인 "DMI ADX Confirmation"
    "Vortex Index": "Vortex Indicator",  # 리딩 "Vortex Index" ↔ 확인 "Vortex Indicator Confirmation"
}

# ---------------------------------------------------------------------------
# 리딩 지표 파라미터 프리셋 (기본값 포함 2~3개 변형)
# 각 프리셋은 기본값 대비 유의미한 범위의 파라미터 조합
# ---------------------------------------------------------------------------
LEADING_PARAM_PRESETS: dict[str, list[dict]] = {
    "Range Filter": [
        {},  # 기본값: period=100, mult=3.0
        {"period": 50, "mult": 2.0},
        {"period": 150, "mult": 4.0},
    ],
    "RQK": [
        {},  # 기본값: lookback=8, relative_weight=8
        {"lookback": 5, "relative_weight": 5},
        {"lookback": 14, "relative_weight": 12},
    ],
    "Supertrend": [
        {},  # 기본값: atr_period=10, factor=3.0
        {"atr_period": 7, "factor": 2.0},
        {"atr_period": 14, "factor": 4.0},
    ],
    "Half Trend": [
        {},  # 기본값: amplitude=2, channel_deviation=2
        {"amplitude": 1, "channel_deviation": 1},
        {"amplitude": 3, "channel_deviation": 3},
    ],
    "Ichimoku Cloud": [
        {},  # 기본값: tenkan=9, kijun=26, senkou=52
        {"tenkan": 7, "kijun": 22, "senkou": 44},
        {"tenkan": 12, "kijun": 30, "senkou": 60},
    ],
    "Chandelier Exit": [
        {},  # 기본값: atr_period=22, mult=3.0
        {"atr_period": 14, "mult": 2.0},
        {"atr_period": 28, "mult": 4.0},
    ],
    "Parabolic SAR": [
        {},  # 기본값: start=0.02, increment=0.02, max=0.2
        {"start": 0.01, "increment": 0.01, "max": 0.1},
        {"start": 0.03, "increment": 0.03, "max": 0.3},
    ],
    "SSL Channel": [
        {},  # 기본값: period=10
        {"period": 7},
        {"period": 14},
    ],
    "Hull Suite": [
        {},  # 기본값: length=55
        {"length": 34},
        {"length": 89},
    ],
    "Donchian Trend Ribbon": [
        {},  # 기본값: period=15
        {"period": 10},
        {"period": 20},
    ],
    "Trendline Breakout": [
        {},  # 기본값: length=14
        {"length": 10},
        {"length": 20},
    ],
    "TSI": [
        {},  # 기본값: long=25, short=13, signal=13
        {"long": 20, "short": 10, "signal": 10},
        {"long": 30, "short": 15, "signal": 15},
    ],
    "TDFI": [
        {},  # 기본값: lookback=13, mma=13, filter_high=0.05
        {"lookback": 8, "mma": 8, "filter_high": 0.03},
        {"lookback": 21, "mma": 21, "filter_high": 0.08},
    ],
    "Stochastic": [
        {},  # 기본값: length=14, smooth_k=3, smooth_d=3
        {"length": 9, "smooth_k": 3, "smooth_d": 3},
        {"length": 21, "smooth_k": 5, "smooth_d": 5},
    ],
    "RSI": [
        {},  # 기본값: length=14, ma_length=14
        {"length": 9, "ma_length": 9},
        {"length": 21, "ma_length": 21},
    ],
    "ROC": [
        {},  # 기본값: length=9
        {"length": 6},
        {"length": 14},
    ],
    "CCI": [
        {},  # 기본값: length=20, upper=100, lower=-100
        {"length": 14, "upper": 100, "lower": -100},
        {"length": 30, "upper": 150, "lower": -150},
    ],
    "MACD": [
        {},  # 기본값: fast=12, slow=26, signal=9
        {"fast": 8, "slow": 21, "signal": 5},
        {"fast": 16, "slow": 36, "signal": 12},
    ],
    "Bull Bear Power Trend": [
        {},  # 기본값: period=50, atr=5
        {"period": 30, "atr": 3},
        {"period": 70, "atr": 7},
    ],
    "DPO": [
        {},  # 기본값: period=10
        {"period": 7},
        {"period": 14},
    ],
    "BB Oscillator": [
        {},  # 기본값: length=20, stddev=2.0
        {"length": 14, "stddev": 1.5},
        {"length": 30, "stddev": 2.5},
    ],
    "Awesome Oscillator": [
        {},  # 기본값: fast=5, slow=34
        {"fast": 3, "slow": 21},
        {"fast": 7, "slow": 55},
    ],
    "Volatility Oscillator": [
        {},  # 기본값: length=100
        {"length": 50},
        {"length": 150},
    ],
    "2 EMA Cross": [
        {},  # 기본값: fast=50, slow=200
        {"fast": 20, "slow": 100},
        {"fast": 50, "slow": 100},
    ],
    "3 EMA Cross": [
        {},  # 기본값: ema1=9, ema2=21, ema3=55
        {"ema1": 5, "ema2": 13, "ema3": 34},
        {"ema1": 12, "ema2": 26, "ema3": 89},
    ],
    "VWAP": [
        {},  # 기본값: length=20
        {"length": 14},
        {"length": 30},
    ],
    "DMI": [
        {},  # 기본값: length=14, adx_smoothing=14
        {"length": 10, "adx_smoothing": 10},
        {"length": 20, "adx_smoothing": 20},
    ],
    "Waddah Attar Explosion": [
        {},  # 기본값: sensitivity=150, fast=20, slow=40, channel_length=20, mult=2.0
        {"sensitivity": 100, "fast": 15, "slow": 30, "channel_length": 15, "mult": 1.5},
        {"sensitivity": 200, "fast": 25, "slow": 50, "channel_length": 25, "mult": 2.5},
    ],
    "Chaikin Money Flow": [
        {},  # 기본값: length=20
        {"length": 14},
        {"length": 30},
    ],
    "Vortex Index": [
        {},  # 기본값: period=14
        {"period": 10},
        {"period": 21},
    ],
    "STC": [
        {},  # 기본값: fast=23, slow=50, cycle=10
        {"fast": 15, "slow": 40, "cycle": 8},
        {"fast": 30, "slow": 60, "cycle": 12},
    ],
    "Range Detector": [
        {},  # 기본값: length=20, mult=1.0, atr_len=500
        {"length": 14, "mult": 0.8, "atr_len": 300},
        {"length": 30, "mult": 1.5, "atr_len": 700},
    ],
    "QQE Mod": [
        {},  # 기본값: rsi_period=6, sf=5, qqe_factor=3
        {"rsi_period": 4, "sf": 3, "qqe_factor": 2},
        {"rsi_period": 10, "sf": 7, "qqe_factor": 4},
    ],
}


@dataclass
class IndicatorCombination:
    """지표 조합 정의 데이터 클래스.

    리딩 지표 1개와 확인 지표 0~N개의 조합을 표현한다.
    """

    id: str  # 고유 식별자 (예: "RangeFilter_conf-EMAFilter+RSI-RSIMACross")
    leading: str  # 리딩 지표 이름 (예: "Range Filter")
    leading_params: dict = field(default_factory=dict)  # 리딩 지표 파라미터
    confirmations: list[dict] = field(default_factory=list)  # 확인 지표 리스트 [{name, subtype, params}]
    signal_expiry: int | None = None  # 시그널 만료 캔들 수 (None이면 config 기본값 사용)


@dataclass
class _ExpandedConfirmation:
    """서브타입이 확장된 개별 확인 지표 정보 (내부용).

    서브타입이 있는 확인 지표를 별도 항목으로 분리하여 조합 생성에 사용한다.
    """

    name: str  # 확인 지표 이름 (예: "TSI Confirmation")
    subtype: str | None  # 서브타입 이름 또는 None
    short_label: str  # ID 생성용 축약 라벨 (예: "TSI-SignalCross")
    base_name: str  # 분류 매핑용 base name (예: "TSI")
    category: str | None  # 실무형 분류 카테고리 (예: "모멘텀") 또는 None


def _conf_base_name(conf_name: str) -> str:
    """확인 지표 이름에서 base name을 추출한다.

    " Confirmation" 접미사가 있으면 제거하여 분류 매핑에 사용할 base name을 반환한다.

    Args:
        conf_name: 확인 지표 전체 이름 (예: "TSI Confirmation", "EMA Filter")

    Returns:
        base name (예: "TSI", "EMA Filter")
    """
    suffix = " Confirmation"
    if conf_name.endswith(suffix):
        return conf_name.removesuffix(suffix)
    return conf_name


def _get_conf_base_for_leading(leading_name: str) -> str:
    """리딩 지표 이름에 대응하는 확인 지표 base name을 반환한다.

    대부분 리딩 이름 == 확인 base name이지만, 일부 불일치가 있어
    LEADING_TO_CONF_BASE 매핑을 우선 참조한다.

    Args:
        leading_name: 리딩 지표 이름 (예: "DMI", "Range Filter")

    Returns:
        대응하는 확인 지표 base name (예: "DMI ADX", "Range Filter")
    """
    return LEADING_TO_CONF_BASE.get(leading_name, leading_name)


def _format_param_suffix(params: dict) -> str:
    """파라미터 딕셔너리를 ID 접미사 문자열로 변환한다.

    빈 딕셔너리(기본값)이면 빈 문자열을 반환한다.

    Args:
        params: 파라미터 딕셔너리 (예: {"period": 50, "mult": 2.0})

    Returns:
        ID 접미사 (예: "_p50-m2.0") 또는 빈 문자열
    """
    if not params:
        return ""
    parts: list[str] = []
    for key, val in params.items():
        # 키의 첫 글자를 축약으로 사용
        abbr = key[0]
        # float이 정수값이면 정수로 표시
        if isinstance(val, float) and val == int(val):
            parts.append(f"{abbr}{int(val)}")
        else:
            parts.append(f"{abbr}{val}")
    return "_" + "-".join(parts)


class CombinationEngine:
    """지표 조합 생성 및 관리 엔진.

    리딩 지표 이름 목록과 확인 지표 정보(이름 + 서브타입)를 받아
    모든 가능한 리딩+확인 지표 조합을 생성한다.

    필터링 규칙:
    - excluded_names에 포함된 지표는 리딩/확인 모두에서 제거
    - 리딩 지표와 동일 계열 확인 지표는 같은 조합에 포함 불가
    - 확인 지표 중 동일 실무형 분류에서 2개 이상 포함 불가
    - 리딩 지표 파라미터 프리셋으로 파라미터 변형 조합 생성
    """

    def __init__(
        self,
        leading_names: list[str],
        confirmation_info: list[dict],
        *,
        excluded_names: set[str] | None = None,
        category_map: dict[str, str] | None = None,
        param_presets: dict[str, list[dict]] | None = None,
    ) -> None:
        """CombinationEngine 초기화.

        Args:
            leading_names: 사용 가능한 리딩 지표 이름 목록
                (예: ["Range Filter", "RQK", "Supertrend", ...])
            confirmation_info: 확인 지표 정보 리스트.
                각 항목은 {"name": str, "subtypes": list[str]} 형태.
                서브타입이 없으면 "subtypes"는 빈 리스트.
            excluded_names: 제외할 지표 base name 집합.
                None이면 EXCLUDED_INDICATORS 사용.
            category_map: 확인 지표 base name → 카테고리 매핑.
                None이면 CONFIRMATION_CATEGORIES 사용.
            param_presets: 리딩 지표 이름 → 파라미터 프리셋 리스트 매핑.
                None이면 LEADING_PARAM_PRESETS 사용.
        """
        self._excluded = excluded_names if excluded_names is not None else EXCLUDED_INDICATORS
        self._category_map = category_map if category_map is not None else CONFIRMATION_CATEGORIES
        self._param_presets = param_presets if param_presets is not None else LEADING_PARAM_PRESETS

        # 제외 지표 필터링 적용
        self._leading_names = [
            n for n in leading_names if n not in self._excluded
        ]  # 필터링된 리딩 지표 이름 목록
        self._confirmation_info = [
            info for info in confirmation_info
            if _conf_base_name(info["name"]) not in self._excluded
        ]  # 필터링된 확인 지표 원본 정보
        self._expanded: list[_ExpandedConfirmation] = self._expand_confirmations()  # 서브타입 확장된 확인 지표 목록

    def _expand_confirmations(self) -> list[_ExpandedConfirmation]:
        """확인 지표의 서브타입을 별도 항목으로 확장한다.

        서브타입이 있는 지표는 각 서브타입별로 분리하고,
        서브타입이 없는 지표는 subtype=None으로 1개 항목을 생성한다.
        각 항목에 base_name과 category 정보를 함께 저장한다.

        Returns:
            서브타입이 확장된 _ExpandedConfirmation 리스트.
        """
        expanded: list[_ExpandedConfirmation] = []
        for info in self._confirmation_info:
            name = info["name"]
            subtypes = info.get("subtypes", [])
            base_label = self._make_short_label(name)
            base_name = _conf_base_name(name)
            category = self._category_map.get(base_name)

            if subtypes:
                for st in subtypes:
                    st_label = st.replace(" ", "").replace("&", "And").replace("%", "Pct")
                    short_label = f"{base_label}-{st_label}"
                    expanded.append(
                        _ExpandedConfirmation(
                            name=name,
                            subtype=st,
                            short_label=short_label,
                            base_name=base_name,
                            category=category,
                        )
                    )
            else:
                expanded.append(
                    _ExpandedConfirmation(
                        name=name,
                        subtype=None,
                        short_label=base_label,
                        base_name=base_name,
                        category=category,
                    )
                )
        return expanded

    @staticmethod
    def _make_short_label(name: str) -> str:
        """지표 이름에서 ID용 축약 라벨을 생성한다.

        공백을 제거하고 'Confirmation' 접미사를 제거하여 간결한 라벨을 만든다.

        Args:
            name: 지표 전체 이름 (예: "TSI Confirmation")

        Returns:
            축약 라벨 (예: "TSI")
        """
        label = name.replace(" Confirmation", "").replace(" ", "")
        return label

    def _is_valid_combination(
        self,
        leading_name: str,
        chosen_indices: tuple[int, ...],
    ) -> bool:
        """선택된 확인 지표 조합이 필터링 규칙을 만족하는지 검증한다.

        규칙:
        1. 리딩 지표와 동일 계열 확인 지표가 포함되면 안 됨
        2. 동일 실무형 분류에서 2개 이상 확인 지표가 포함되면 안 됨

        Args:
            leading_name: 리딩 지표 이름
            chosen_indices: 선택된 확인 지표의 _expanded 인덱스 튜플

        Returns:
            유효한 조합이면 True, 아니면 False
        """
        if not chosen_indices:
            return True

        # 리딩 지표에 대응하는 확인 지표 base name
        leading_conf_base = _get_conf_base_for_leading(leading_name)

        category_count: dict[str, int] = {}  # 카테고리별 확인 지표 수

        for idx in chosen_indices:
            ec = self._expanded[idx]

            # 규칙 1: 리딩-확인 겹침 방지
            if ec.base_name == leading_conf_base:
                return False

            # 규칙 2: 동일 분류 2개 이상 방지
            cat = ec.category
            if cat is not None:
                category_count[cat] = category_count.get(cat, 0) + 1
                if category_count[cat] > 1:
                    return False

        return True

    def _estimate_total(self, max_confirmations: int) -> int:
        """주어진 max_confirmations에 대한 총 조합 수를 추정한다.

        필터링 규칙 적용 전의 상한 추정치를 반환한다.
        공식: L × P × Σ(k=0..N) C(C_count, k)
        여기서 L=리딩 지표 수, P=평균 파라미터 프리셋 수,
        C_count=확장된 확인 지표 수, N=max_confirmations.

        Args:
            max_confirmations: 최대 확인 지표 수 (0 이상)

        Returns:
            추정 총 조합 수 (상한).
        """
        l_count = len(self._leading_names)  # 리딩 지표 수
        c_count = len(self._expanded)  # 확장된 확인 지표 수
        n = min(max_confirmations, c_count)  # 실제 적용 가능한 최대 확인 지표 수
        confirmation_combos = sum(comb(c_count, k) for k in range(n + 1))

        # 파라미터 프리셋 수 반영 (평균)
        total_presets = sum(
            len(self._param_presets.get(name, [{}]))
            for name in self._leading_names
        )
        avg_presets = total_presets / l_count if l_count > 0 else 1

        return int(l_count * avg_presets * confirmation_combos)

    def generate(
        self,
        max_confirmations: int = 3,
        max_combinations: int = 100_000,
        signal_expiry_values: list[int] | None = None,
    ) -> list[IndicatorCombination]:
        """모든 가능한 리딩+확인 지표 조합을 생성한다.

        필터링 규칙을 적용하여 유효한 조합만 생성한다:
        - 제외 지표 제거 (생성자에서 이미 적용)
        - 리딩-확인 겹침 방지
        - 동일 분류 2개 이상 확인 지표 방지
        - 리딩 지표 파라미터 프리셋 변형 포함
        - signal_expiry 값별 조합 복제 (확인 지표가 있는 조합만)

        조합 수가 max_combinations를 초과하면 max_confirmations를
        자동으로 축소하여 한도 이내로 조정한다.

        Args:
            max_confirmations: 최대 확인 지표 수 (기본값: 3)
            max_combinations: 최대 조합 수 한도 (기본값: 100,000)
            signal_expiry_values: 시그널 만료 캔들 수 변형 리스트
                (예: [1, 2, 3, 5, 7]). None이면 signal_expiry 변형 없이 기존 동작.

        Returns:
            생성된 IndicatorCombination 리스트.
        """
        # 조합 수 초과 시 max_confirmations 자동 축소
        current_max = max_confirmations
        while current_max > 0 and self._estimate_total(current_max) > max_combinations:
            current_max -= 1

        base_combos: list[IndicatorCombination] = []
        c_count = len(self._expanded)
        n = min(current_max, c_count)

        for leading_name in self._leading_names:
            leading_label = self._make_short_label(leading_name)
            presets = self._param_presets.get(leading_name, [{}])

            for params in presets:
                param_suffix = _format_param_suffix(params)

                # k=0: 확인 지표 없는 조합
                for k in range(n + 1):
                    if k == 0:
                        combo_id = f"{leading_label}{param_suffix}"
                        base_combos.append(
                            IndicatorCombination(
                                id=combo_id,
                                leading=leading_name,
                                leading_params=dict(params),
                                confirmations=[],
                            )
                        )
                    else:
                        for chosen in combinations(range(c_count), k):
                            # 필터링 규칙 검증
                            if not self._is_valid_combination(leading_name, chosen):
                                continue

                            conf_list: list[dict] = []
                            conf_labels: list[str] = []
                            for idx in chosen:
                                ec = self._expanded[idx]
                                conf_entry: dict = {"name": ec.name, "params": {}}
                                if ec.subtype is not None:
                                    conf_entry["subtype"] = ec.subtype
                                conf_list.append(conf_entry)
                                conf_labels.append(ec.short_label)

                            conf_part = "+".join(conf_labels)
                            combo_id = f"{leading_label}{param_suffix}_conf-{conf_part}"
                            base_combos.append(
                                IndicatorCombination(
                                    id=combo_id,
                                    leading=leading_name,
                                    leading_params=dict(params),
                                    confirmations=conf_list,
                                )
                            )

        # signal_expiry 변형 적용: 확인 지표가 있는 조합만 복제
        result: list[IndicatorCombination] = []
        if signal_expiry_values and len(signal_expiry_values) > 1:
            for combo in base_combos:
                if combo.confirmations:
                    # 확인 지표가 있는 조합: signal_expiry 값별로 복제
                    for se in signal_expiry_values:
                        result.append(
                            IndicatorCombination(
                                id=f"{combo.id}|se={se}",
                                leading=combo.leading,
                                leading_params=dict(combo.leading_params),
                                confirmations=list(combo.confirmations),
                                signal_expiry=se,
                            )
                        )
                else:
                    # 확인 지표 없는 조합: signal_expiry 무관하므로 그대로 유지
                    result.append(combo)
        else:
            result = base_combos

        total = len(result)
        se_info = f", signal_expiry={signal_expiry_values}" if signal_expiry_values else ""
        print(f"총 조합 수: {total} (max_confirmations={current_max}, 필터링 적용{se_info})")

        return result
