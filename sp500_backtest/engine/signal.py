"""매매 시그널 생성기 모듈.

리딩 지표 시그널과 확인 지표 필터를 결합하여
최종 포지션 시그널(Long/Short/무포지션)을 생성한다.
Signal Expiry 및 Alternate Signal 로직을 포함한다.
"""

import numpy as np
import pandas as pd

from sp500_backtest.indicators.base import IndicatorResult


class SignalGenerator:
    """매매 시그널 생성기.

    리딩 지표 1개와 확인 지표 0~N개의 시그널을 결합하여
    최종 포지션 시그널을 생성한다.
    """

    def generate(
        self,
        leading_result: IndicatorResult,
        confirmation_results: list[IndicatorResult],
        signal_expiry: int = 3,
        alternate_signal: bool = True,
    ) -> pd.Series:
        """리딩 지표 시그널과 확인 지표 필터를 결합하여 최종 포지션 시그널을 생성한다.

        Signal Expiry 로직:
        - 리딩 지표 시그널 발생 후 signal_expiry 캔들 이내에
          모든 확인 지표가 확인되지 않으면 시그널 무효화

        Alternate Signal 로직:
        - 연속 동일 방향 시그널 필터링 (Long 후 다시 Long 무시)

        Args:
            leading_result: 리딩 지표 계산 결과 (long_signal, short_signal).
            confirmation_results: 확인 지표 계산 결과 리스트.
            signal_expiry: 시그널 만료 캔들 수 (기본값: 3).
            alternate_signal: 연속 동일 방향 시그널 필터링 여부 (기본값: True).

        Returns:
            pd.Series: 1(Long), -1(Short), 0(무포지션) 값의 시그널 시리즈.
        """
        index = leading_result.long_signal.index
        n = len(index)

        if n == 0:
            return pd.Series(dtype=np.int64)

        # 1단계: 확인 지표가 없으면 리딩 시그널을 직접 사용
        if not confirmation_results:
            raw_signal = self._combine_without_confirmations(leading_result)
        else:
            raw_signal = self._combine_with_expiry(
                leading_result, confirmation_results, signal_expiry
            )

        # 2단계: Alternate Signal 필터링
        if alternate_signal:
            raw_signal = self._apply_alternate_signal(raw_signal)

        return raw_signal

    def _combine_without_confirmations(
        self, leading_result: IndicatorResult
    ) -> pd.Series:
        """확인 지표 없이 리딩 시그널만으로 포지션 시그널을 생성한다.

        Args:
            leading_result: 리딩 지표 계산 결과.

        Returns:
            pd.Series: 1(Long), -1(Short), 0(무포지션).
        """
        long_sig = leading_result.long_signal.fillna(False).astype(bool)
        short_sig = leading_result.short_signal.fillna(False).astype(bool)

        signal = pd.Series(0, index=long_sig.index, dtype=np.int64)
        signal[long_sig] = 1
        signal[short_sig] = -1

        return signal

    def _combine_with_expiry(
        self,
        leading_result: IndicatorResult,
        confirmation_results: list[IndicatorResult],
        signal_expiry: int,
    ) -> pd.Series:
        """Signal Expiry 로직을 적용하여 리딩+확인 시그널을 결합한다.

        리딩 지표가 시그널을 발생시키면 signal_expiry 캔들 이내에
        모든 확인 지표가 동일 방향을 확인해야 유효한 시그널이 된다.

        Args:
            leading_result: 리딩 지표 계산 결과.
            confirmation_results: 확인 지표 계산 결과 리스트.
            signal_expiry: 시그널 만료 캔들 수.

        Returns:
            pd.Series: 1(Long), -1(Short), 0(무포지션).
        """
        index = leading_result.long_signal.index
        n = len(index)

        leading_long = leading_result.long_signal.fillna(False).values.astype(bool)
        leading_short = leading_result.short_signal.fillna(False).values.astype(bool)

        # 확인 지표 long/short 배열 (num_confirmations × n)
        conf_longs = np.array(
            [r.long_signal.fillna(False).values.astype(bool) for r in confirmation_results]
        )
        conf_shorts = np.array(
            [r.short_signal.fillna(False).values.astype(bool) for r in confirmation_results]
        )

        # 모든 확인 지표가 동시에 확인하는 시점 (AND 조건)
        all_conf_long = conf_longs.all(axis=0)  # shape: (n,)
        all_conf_short = conf_shorts.all(axis=0)  # shape: (n,)

        signal = np.zeros(n, dtype=np.int64)

        # 리딩 시그널 발생 시점 추적 및 expiry 윈도우 내 확인 검사
        # Long 시그널 처리
        long_pending = False  # 리딩 Long 시그널 대기 중 여부
        long_fire_idx = -1  # 리딩 Long 시그널 발생 인덱스

        short_pending = False  # 리딩 Short 시그널 대기 중 여부
        short_fire_idx = -1  # 리딩 Short 시그널 발생 인덱스

        for i in range(n):
            # 리딩 Long 시그널 발생
            if leading_long[i]:
                # 즉시 확인 가능한지 체크
                if all_conf_long[i]:
                    signal[i] = 1
                    long_pending = False
                else:
                    long_pending = True
                    long_fire_idx = i

            # 리딩 Short 시그널 발생
            if leading_short[i]:
                # 즉시 확인 가능한지 체크
                if all_conf_short[i]:
                    signal[i] = -1
                    short_pending = False
                else:
                    short_pending = True
                    short_fire_idx = i

            # 대기 중인 Long 시그널의 expiry 윈도우 내 확인 검사
            if long_pending and not leading_long[i]:
                if (i - long_fire_idx) <= signal_expiry:
                    if all_conf_long[i]:
                        signal[i] = 1
                        long_pending = False
                else:
                    # 만료: 윈도우 초과
                    long_pending = False

            # 대기 중인 Short 시그널의 expiry 윈도우 내 확인 검사
            if short_pending and not leading_short[i]:
                if (i - short_fire_idx) <= signal_expiry:
                    if all_conf_short[i]:
                        signal[i] = -1
                        short_pending = False
                else:
                    # 만료: 윈도우 초과
                    short_pending = False

        return pd.Series(signal, index=index, dtype=np.int64)

    @staticmethod
    def _apply_alternate_signal(signal: pd.Series) -> pd.Series:
        """연속 동일 방향 시그널을 필터링한다.

        Long(+1) 후 다시 Long이 오면 무시(0), Short(-1) 후 다시 Short이 오면 무시(0).
        0이 아닌 시그널만 추출했을 때 인접한 두 값은 항상 부호가 달라야 한다.

        Args:
            signal: 원본 시그널 시리즈 (1, -1, 0).

        Returns:
            pd.Series: 필터링된 시그널 시리즈.
        """
        result = signal.copy()
        last_direction = 0  # 마지막으로 유효했던 시그널 방향

        for i in range(len(result)):
            val = result.iloc[i]
            if val != 0:
                if val == last_direction:
                    # 연속 동일 방향 → 무효화
                    result.iloc[i] = 0
                else:
                    last_direction = val

        return result
