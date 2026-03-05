"""지표 조합 생성 엔진 모듈.

37개 리딩 지표와 확인 지표(서브타입 포함)의 모든 가능한 조합을
생성하고 관리한다. 조합 수가 상한을 초과하면 max_confirmations를
자동으로 축소한다.
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import comb


@dataclass
class IndicatorCombination:
    """지표 조합 정의 데이터 클래스.

    리딩 지표 1개와 확인 지표 0~N개의 조합을 표현한다.
    """

    id: str  # 고유 식별자 (예: "RangeFilter_conf-EMAFilter+RSI-RSIMACross")
    leading: str  # 리딩 지표 이름 (예: "Range Filter")
    leading_params: dict = field(default_factory=dict)  # 리딩 지표 파라미터
    confirmations: list[dict] = field(default_factory=list)  # 확인 지표 리스트 [{name, subtype, params}]


@dataclass
class _ExpandedConfirmation:
    """서브타입이 확장된 개별 확인 지표 정보 (내부용).

    서브타입이 있는 확인 지표를 별도 항목으로 분리하여 조합 생성에 사용한다.
    """

    name: str  # 확인 지표 이름 (예: "TSI Confirmation")
    subtype: str | None  # 서브타입 이름 또는 None
    short_label: str  # ID 생성용 축약 라벨 (예: "TSI-SignalCross")


class CombinationEngine:
    """지표 조합 생성 및 관리 엔진.

    리딩 지표 이름 목록과 확인 지표 정보(이름 + 서브타입)를 받아
    모든 가능한 리딩+확인 지표 조합을 생성한다.
    """

    def __init__(
        self,
        leading_names: list[str],
        confirmation_info: list[dict],
    ) -> None:
        """CombinationEngine 초기화.

        Args:
            leading_names: 사용 가능한 리딩 지표 이름 목록
                (예: ["Range Filter", "RQK", "Supertrend", ...])
            confirmation_info: 확인 지표 정보 리스트.
                각 항목은 {"name": str, "subtypes": list[str]} 형태.
                서브타입이 없으면 "subtypes"는 빈 리스트.
                (예: [{"name": "TSI Confirmation", "subtypes": ["Signal Cross", "Zero line cross"]},
                      {"name": "EMA Filter Confirmation", "subtypes": []}])
        """
        self._leading_names = leading_names  # 리딩 지표 이름 목록
        self._confirmation_info = confirmation_info  # 확인 지표 원본 정보
        self._expanded: list[_ExpandedConfirmation] = self._expand_confirmations()  # 서브타입 확장된 확인 지표 목록

    def _expand_confirmations(self) -> list[_ExpandedConfirmation]:
        """확인 지표의 서브타입을 별도 항목으로 확장한다.

        서브타입이 있는 지표는 각 서브타입별로 분리하고,
        서브타입이 없는 지표는 subtype=None으로 1개 항목을 생성한다.

        Returns:
            서브타입이 확장된 _ExpandedConfirmation 리스트.
        """
        expanded: list[_ExpandedConfirmation] = []
        for info in self._confirmation_info:
            name = info["name"]
            subtypes = info.get("subtypes", [])
            base_label = self._make_short_label(name)
            if subtypes:
                for st in subtypes:
                    st_label = st.replace(" ", "").replace("&", "And").replace("%", "Pct")
                    short_label = f"{base_label}-{st_label}"
                    expanded.append(
                        _ExpandedConfirmation(name=name, subtype=st, short_label=short_label)
                    )
            else:
                expanded.append(
                    _ExpandedConfirmation(name=name, subtype=None, short_label=base_label)
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

    def _estimate_total(self, max_confirmations: int) -> int:
        """주어진 max_confirmations에 대한 총 조합 수를 추정한다.

        공식: L × Σ(k=0..N) C(C_count, k)
        여기서 L=리딩 지표 수, C_count=확장된 확인 지표 수, N=max_confirmations.

        Args:
            max_confirmations: 최대 확인 지표 수 (0 이상)

        Returns:
            추정 총 조합 수.
        """
        l_count = len(self._leading_names)  # 리딩 지표 수
        c_count = len(self._expanded)  # 확장된 확인 지표 수
        n = min(max_confirmations, c_count)  # 실제 적용 가능한 최대 확인 지표 수
        confirmation_combos = sum(comb(c_count, k) for k in range(n + 1))
        return l_count * confirmation_combos

    def generate(
        self,
        max_confirmations: int = 3,
        max_combinations: int = 100_000,
    ) -> list[IndicatorCombination]:
        """모든 가능한 리딩+확인 지표 조합을 생성한다.

        조합 수가 max_combinations를 초과하면 max_confirmations를
        자동으로 축소하여 한도 이내로 조정한다.

        Args:
            max_confirmations: 최대 확인 지표 수 (기본값: 3)
            max_combinations: 최대 조합 수 한도 (기본값: 100,000)

        Returns:
            생성된 IndicatorCombination 리스트.
        """
        # 조합 수 초과 시 max_confirmations 자동 축소
        current_max = max_confirmations
        while current_max > 0 and self._estimate_total(current_max) > max_combinations:
            current_max -= 1

        total = self._estimate_total(current_max)
        print(f"총 조합 수: {total} (max_confirmations={current_max})")

        result: list[IndicatorCombination] = []
        c_count = len(self._expanded)
        n = min(current_max, c_count)

        for leading_name in self._leading_names:
            leading_label = self._make_short_label(leading_name)

            # k=0: 확인 지표 없는 조합
            for k in range(n + 1):
                if k == 0:
                    combo_id = leading_label
                    result.append(
                        IndicatorCombination(
                            id=combo_id,
                            leading=leading_name,
                            leading_params={},
                            confirmations=[],
                        )
                    )
                else:
                    for chosen in combinations(range(c_count), k):
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
                        combo_id = f"{leading_label}_conf-{conf_part}"
                        result.append(
                            IndicatorCombination(
                                id=combo_id,
                                leading=leading_name,
                                leading_params={},
                                confirmations=conf_list,
                            )
                        )

        return result
