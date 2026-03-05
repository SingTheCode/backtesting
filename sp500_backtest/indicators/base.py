"""지표 기본 인터페이스 모듈.

IndicatorResult 데이터 모델, BaseIndicator 추상 클래스,
ConfirmationIndicator 서브타입 지원 클래스를 정의한다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class IndicatorResult:
    """지표 계산 결과를 담는 데이터 클래스.

    모든 지표(리딩/확인)의 calculate() 메서드가 반환하는 표준 결과 형식이다.
    """

    long_signal: pd.Series  # 매수(Long) 시그널 (True=진입, False=미진입)
    short_signal: pd.Series  # 매도(Short) 시그널 (True=진입, False=미진입)


class BaseIndicator(ABC):
    """모든 지표의 기본 추상 클래스.

    37개 리딩 지표와 45개+ 확인 지표가 공통으로 상속하는 인터페이스를 정의한다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """지표 이름 (예: 'RSI', 'Supertrend', 'MACD')."""
        ...

    @property
    @abstractmethod
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터 딕셔너리 (예: {'length': 14, 'ma_length': 14})."""
        ...

    @abstractmethod
    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """지표를 계산하여 Long/Short 시그널을 반환한다.

        Args:
            df: OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
            params: 파라미터 딕셔너리. None이면 default_params 사용.

        Returns:
            IndicatorResult: long_signal, short_signal boolean Series.
        """
        ...

    def _resolve_params(self, params: dict | None) -> dict:
        """전달된 파라미터와 기본 파라미터를 병합한다.

        Args:
            params: 사용자 지정 파라미터. None이면 기본값만 사용.

        Returns:
            기본값에 사용자 지정값을 덮어쓴 최종 파라미터 딕셔너리.
        """
        resolved = dict(self.default_params)
        if params:
            resolved.update(params)
        return resolved


class ConfirmationIndicator(BaseIndicator):
    """확인 지표 기본 클래스 (서브타입 지원).

    서브타입이 있는 확인 지표(예: TSI의 Signal Cross/Zero line cross)를 위한
    확장 인터페이스를 제공한다.
    """

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록 (예: ['Signal Cross', 'Zero line cross']).

        서브타입이 없는 지표는 빈 리스트를 반환한다.
        """
        return []

    def calculate(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        params: dict | None = None,
        subtype: str | None = None,
    ) -> IndicatorResult:
        """서브타입을 포함한 지표 계산.

        Args:
            df: OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
            params: 파라미터 딕셔너리. None이면 default_params 사용.
            subtype: 서브타입 이름. None이면 기본 동작 수행.

        Returns:
            IndicatorResult: long_signal, short_signal boolean Series.

        Raises:
            ValueError: 지원하지 않는 subtype이 전달된 경우.
        """
        if subtype is not None and self.subtypes and subtype not in self.subtypes:
            raise ValueError(
                f"'{self.name}' 지표는 '{subtype}' 서브타입을 지원하지 않습니다. "
                f"지원 서브타입: {self.subtypes}"
            )
        return self._calculate_impl(df, self._resolve_params(params), subtype)

    @abstractmethod
    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """서브클래스에서 구현할 실제 계산 로직.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터 딕셔너리.
            subtype: 서브타입 이름 또는 None.

        Returns:
            IndicatorResult: long_signal, short_signal boolean Series.
        """
        ...
