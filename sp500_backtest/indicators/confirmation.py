"""확인 지표 모듈 (Confirmation Indicators).

리딩 지표의 시그널을 필터링하여 확인하는 보조 지표를 구현한다.
확인 지표는 리딩 지표와 달리 STATE 기반 시그널을 반환한다:
- 리딩 지표: 크로스오버 시점에서만 True (전환 감지)
- 확인 지표: 조건이 유지되는 동안 계속 True (상태 감지)
"""

import numpy as np
import pandas as pd

from sp500_backtest.indicators.base import ConfirmationIndicator, IndicatorResult
from sp500_backtest.indicators.utils import (
    atr,
    crossover,
    crossunder,
    ma,
    rsi as rsi_func,
    stoch as stoch_func,
)


class EMAFilterConfirmation(ConfirmationIndicator):
    """EMA 필터 확인 지표.

    종가가 EMA 위에 있으면 Long, 아래에 있으면 Short 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "EMA Filter"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 200,  # EMA 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """EMA 필터 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: 종가 > EMA이면 Long, 종가 < EMA이면 Short.
        """
        length = int(params["length"])  # EMA 기간
        close = df["Close"]  # 종가 시계열

        ema_val = ma(close, length, "EMA")  # EMA 계산

        long_signal = close > ema_val  # 종가가 EMA 위 (상승 상태)
        short_signal = close < ema_val  # 종가가 EMA 아래 (하락 상태)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class TwoEMACrossConfirmation(ConfirmationIndicator):
    """2 EMA Cross 확인 지표.

    빠른 EMA가 느린 EMA 위에 있으면 Long, 아래에 있으면 Short 상태를 반환한다.
    리딩 지표와 달리 크로스오버 시점이 아닌 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "2 EMA Cross Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 50,  # 빠른 EMA 기간
            "slow": 200,  # 느린 EMA 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """2 EMA Cross 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: fast EMA > slow EMA이면 Long 상태.
        """
        fast = int(params["fast"])  # 빠른 EMA 기간
        slow = int(params["slow"])  # 느린 EMA 기간
        close = df["Close"]  # 종가 시계열

        fast_ema = ma(close, fast, "EMA")  # 빠른 EMA
        slow_ema = ma(close, slow, "EMA")  # 느린 EMA

        long_signal = fast_ema > slow_ema  # 빠른 EMA가 느린 EMA 위 (상승 상태)
        short_signal = fast_ema < slow_ema  # 빠른 EMA가 느린 EMA 아래 (하락 상태)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ThreeEMACrossConfirmation(ConfirmationIndicator):
    """3 EMA Cross 확인 지표.

    3개 EMA가 강세 정렬(ema1 > ema2 > ema3)이면 Long,
    약세 정렬(ema1 < ema2 < ema3)이면 Short 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "3 EMA Cross Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "ema1": 9,  # 가장 빠른 EMA 기간
            "ema2": 21,  # 중간 EMA 기간
            "ema3": 55,  # 가장 느린 EMA 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """3 EMA Cross 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: ema1 > ema2 > ema3이면 Long 상태.
        """
        ema1_len = int(params["ema1"])  # 가장 빠른 EMA 기간
        ema2_len = int(params["ema2"])  # 중간 EMA 기간
        ema3_len = int(params["ema3"])  # 가장 느린 EMA 기간
        close = df["Close"]  # 종가 시계열

        ema1 = ma(close, ema1_len, "EMA")  # 빠른 EMA
        ema2 = ma(close, ema2_len, "EMA")  # 중간 EMA
        ema3 = ma(close, ema3_len, "EMA")  # 느린 EMA

        long_signal = (ema1 > ema2) & (ema2 > ema3)  # 강세 정렬 상태
        short_signal = (ema1 < ema2) & (ema2 < ema3)  # 약세 정렬 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RangeFilterConfirmation(ConfirmationIndicator):
    """Range Filter 확인 지표.

    smoothrng + rngfilt 함수를 사용하여 필터 방향 상태를 반환한다.
    서브타입: Default (기본 방향), DW (Direction-Weighted 변형).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Range Filter Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 100,  # 평활 기간
            "mult": 3.0,  # 범위 승수
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Default", "DW"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Range Filter 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "Default" 또는 "DW" (Direction-Weighted).

        Returns:
            IndicatorResult: 필터 상승 중이면 Long, 하락 중이면 Short 상태.
        """
        period = int(params["period"])  # 평활 기간
        mult = float(params["mult"])  # 범위 승수
        close = df["Close"]  # 종가 시계열
        n = len(close)  # 데이터 길이

        # smoothrng: EMA(|close - close[1]|, period) * mult
        abs_diff = close.diff().abs()  # |종가 변화량|
        smooth_rng = abs_diff.ewm(span=period, adjust=False).mean() * mult  # 평활 범위

        # rngfilt: 반복 계산 (벡터화 불가, 순차 처리)
        filt = np.full(n, np.nan)  # 필터 값 배열
        close_vals = close.values  # 종가 numpy 배열
        srng_vals = smooth_rng.values  # 평활 범위 numpy 배열

        # 첫 유효 인덱스 찾기
        start_idx = 0  # 시작 인덱스
        for i in range(n):
            if not np.isnan(srng_vals[i]):
                filt[i] = close_vals[i]
                start_idx = i + 1
                break

        for i in range(start_idx, n):
            if np.isnan(srng_vals[i]):
                continue
            prev = filt[i - 1]  # 이전 필터 값
            if np.isnan(prev):
                filt[i] = close_vals[i]
                continue
            rng = srng_vals[i]  # 현재 평활 범위
            if close_vals[i] > prev:
                filt[i] = max(prev, close_vals[i] - rng)
            else:
                filt[i] = min(prev, close_vals[i] + rng)

        filt_series = pd.Series(filt, index=close.index)  # 필터 시리즈
        filt_shifted = filt_series.shift(1)  # 이전 필터 값

        # 방향 상태 판단
        upward = filt_series > filt_shifted  # 필터 상승 중
        downward = filt_series < filt_shifted  # 필터 하락 중

        active_subtype = subtype or "Default"  # 기본 서브타입

        if active_subtype == "DW":
            # DW: 종가가 필터 위면 Long, 아래면 Short (상태 기반)
            long_signal = close > filt_series  # 종가가 필터 위
            short_signal = close < filt_series  # 종가가 필터 아래
        else:
            # Default: 단순 방향 상태
            long_signal = upward  # 필터 상승 상태
            short_signal = downward  # 필터 하락 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RQKConfirmation(ConfirmationIndicator):
    """RQK (Rational Quadratic Kernel) 확인 지표.

    Nadaraya-Watson 커널 회귀 추정값이 상승 중이면 Long,
    하락 중이면 Short 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "RQK Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "lookback": 8,  # 룩백 기간
            "relative_weight": 8,  # 상대 가중치 (alpha)
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """RQK 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: 커널 추정값 상승 중이면 Long 상태.
        """
        lookback = int(params["lookback"])  # 룩백 기간
        alpha = float(params["relative_weight"])  # 상대 가중치
        close = df["Close"]  # 종가 시계열
        n = len(close)  # 데이터 길이
        close_vals = close.values  # 종가 numpy 배열

        # Rational Quadratic Kernel 회귀 계산
        yhat = np.full(n, np.nan)  # 커널 회귀 추정값

        for i in range(lookback, n):
            weights = np.zeros(lookback)  # 커널 가중치
            for j in range(lookback):
                # RQ 커널: (1 + (j+1)^2 / (2 * alpha * lookback^2))^(-alpha)
                dist_sq = (j + 1) ** 2  # 거리 제곱
                weights[j] = (
                    1.0 + dist_sq / (2.0 * alpha * lookback**2)
                ) ** (-alpha)

            w_sum = weights.sum()  # 가중치 합
            if w_sum > 0:
                vals = close_vals[i - lookback : i][::-1]  # 역순 (최근→과거)
                yhat[i] = np.dot(vals, weights) / w_sum  # 가중 평균

        yhat_series = pd.Series(yhat, index=close.index)  # 커널 회귀 시리즈
        yhat_prev = yhat_series.shift(1)  # 이전 커널 회귀 값

        # 상태 기반 시그널 (크로스오버가 아닌 지속 상태)
        long_signal = yhat_series > yhat_prev  # 커널 추정값 상승 중
        short_signal = yhat_series < yhat_prev  # 커널 추정값 하락 중

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class SupertrendConfirmation(ConfirmationIndicator):
    """Supertrend 확인 지표.

    ATR 기반 Supertrend 방향이 상승(direction == -1)이면 Long,
    하락(direction == 1)이면 Short 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Supertrend Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "atr_period": 10,  # ATR 기간
            "factor": 3.0,  # ATR 승수
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Supertrend 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: direction == -1이면 Long 상태 (상승 추세).
        """
        atr_period = int(params["atr_period"])  # ATR 기간
        factor = float(params["factor"])  # ATR 승수

        close = df["Close"].values  # 종가 numpy 배열
        high = df["High"].values  # 고가 numpy 배열
        low = df["Low"].values  # 저가 numpy 배열
        hl2 = (high + low) / 2.0  # 중간가 (High+Low)/2
        n = len(close)  # 데이터 길이

        atr_vals = atr(df, atr_period).values  # ATR 값 배열

        upper_band = hl2 + factor * atr_vals  # 상단 밴드
        lower_band = hl2 - factor * atr_vals  # 하단 밴드

        supertrend = np.full(n, np.nan)  # Supertrend 값
        direction = np.ones(n, dtype=int)  # 방향 (1=하락, -1=상승)

        final_upper = np.copy(upper_band)  # 최종 상단 밴드
        final_lower = np.copy(lower_band)  # 최종 하단 밴드

        for i in range(1, n):
            if np.isnan(atr_vals[i]):
                continue

            # 하단 밴드 조정: 이전 하단보다 높으면 유지
            if lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i - 1]

            # 상단 밴드 조정: 이전 상단보다 낮으면 유지
            if upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i - 1]

            # 방향 결정
            if i == 1:
                direction[i] = 1 if close[i] > final_upper[i] else -1
            else:
                prev_st = supertrend[i - 1]
                if np.isnan(prev_st):
                    direction[i] = 1 if close[i] > final_upper[i] else -1
                elif prev_st == final_upper[i - 1]:
                    direction[i] = -1 if close[i] > final_upper[i] else 1
                else:
                    direction[i] = 1 if close[i] < final_lower[i] else -1

            # Supertrend 값 설정
            if direction[i] == -1:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]

        dir_series = pd.Series(direction, index=df.index)  # 방향 시리즈

        # 상태 기반 시그널 (크로스오버가 아닌 지속 상태)
        long_signal = dir_series == -1  # 상승 추세 상태
        short_signal = dir_series == 1  # 하락 추세 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class HalfTrendConfirmation(ConfirmationIndicator):
    """Half Trend 확인 지표.

    ATR 기반 트렌드 추적에서 trend == 0(상승)이면 Long,
    trend == 1(하락)이면 Short 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Half Trend Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "amplitude": 2,  # 진폭 (ATR 기간)
            "channel_deviation": 2,  # 채널 편차 승수
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Half Trend 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: trend == 0이면 Long 상태 (상승 추세).
        """
        amplitude = int(params["amplitude"])  # 진폭
        channel_dev = float(params["channel_deviation"])  # 채널 편차

        close = df["Close"].values  # 종가
        high = df["High"].values  # 고가
        low = df["Low"].values  # 저가
        n = len(close)  # 데이터 길이

        atr_vals = atr(df, max(amplitude, 1)).values  # ATR 값
        dev = channel_dev * atr_vals  # 편차

        # 고가/저가의 amplitude 기간 최고/최저
        high_ma = (
            pd.Series(high).rolling(window=max(amplitude, 1), min_periods=1).max().values
        )  # 기간 최고가
        low_ma = (
            pd.Series(low).rolling(window=max(amplitude, 1), min_periods=1).min().values
        )  # 기간 최저가

        trend = np.zeros(n, dtype=int)  # 트렌드 방향 (0=상승, 1=하락)
        half_trend = np.full(n, np.nan)  # Half Trend 값

        half_trend[0] = close[0]  # 초기값

        for i in range(1, n):
            if np.isnan(atr_vals[i]):
                half_trend[i] = (
                    half_trend[i - 1] if not np.isnan(half_trend[i - 1]) else close[i]
                )
                trend[i] = trend[i - 1]
                continue

            prev_trend = trend[i - 1]  # 이전 트렌드
            prev_ht = half_trend[i - 1]  # 이전 Half Trend 값

            if prev_trend == 0:  # 상승 트렌드
                max_low = low_ma[i]  # 기간 최저가
                new_ht = max(prev_ht, max_low)  # 상승 시 최저가 추적
                if close[i] < new_ht - dev[i]:
                    trend[i] = 1  # 하락 전환
                    half_trend[i] = high_ma[i]
                else:
                    trend[i] = 0
                    half_trend[i] = new_ht
            else:  # 하락 트렌드
                min_high = high_ma[i]  # 기간 최고가
                new_ht = min(prev_ht, min_high)  # 하락 시 최고가 추적
                if close[i] > new_ht + dev[i]:
                    trend[i] = 0  # 상승 전환
                    half_trend[i] = low_ma[i]
                else:
                    trend[i] = 1
                    half_trend[i] = new_ht

        trend_series = pd.Series(trend, index=df.index)  # 트렌드 시리즈

        # 상태 기반 시그널 (크로스오버가 아닌 지속 상태)
        long_signal = trend_series == 0  # 상승 추세 상태
        short_signal = trend_series == 1  # 하락 추세 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DonchianTrendRibbonConfirmation(ConfirmationIndicator):
    """Donchian Trend Ribbon 확인 지표.

    5개 기간의 Donchian 채널 중간값 대비 종가 위치를 합산하여
    trend_sum > 0이면 Long, trend_sum < 0이면 Short 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Donchian Trend Ribbon Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 15,  # Donchian 채널 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Donchian Trend Ribbon 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: trend_sum > 0이면 Long 상태.
        """
        period = int(params["period"])  # Donchian 채널 기간

        close = df["Close"]  # 종가
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        # 5개 기간의 Donchian 채널 중간값 계산
        dch_periods = [period, period * 2, period * 3, period * 4, period * 5]  # 5개 기간
        trend_sum = pd.Series(0.0, index=df.index)  # 트렌드 합산

        for dp in dch_periods:
            upper = high.rolling(window=dp, min_periods=dp).max()  # 기간 최고가
            lower = low.rolling(window=dp, min_periods=dp).min()  # 기간 최저가
            mid = (upper + lower) / 2.0  # 중간값
            # 종가가 중간값 위면 +1, 아래면 -1
            trend_sum = trend_sum + pd.Series(
                np.where(close > mid, 1.0, np.where(close < mid, -1.0, 0.0)),
                index=df.index,
            )

        # 상태 기반 시그널 (크로스오버가 아닌 지속 상태)
        long_signal = trend_sum > 0  # 상승 트렌드 상태
        short_signal = trend_sum < 0  # 하락 트렌드 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ROCConfirmation(ConfirmationIndicator):
    """ROC (Rate of Change) 확인 지표.

    ROC가 0 위에 있으면 Long, 0 아래에 있으면 Short 상태를 반환한다.
    리딩 지표와 달리 크로스오버 시점이 아닌 상태를 반환한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "ROC Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 9,  # ROC 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """ROC 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: ROC > 0이면 Long 상태.
        """
        length = int(params["length"])  # ROC 기간
        close = df["Close"]  # 종가
        prev_close = close.shift(length)  # length 기간 전 종가

        # ROC = (현재종가 - 과거종가) / 과거종가 * 100
        roc_val = (close - prev_close) / prev_close.replace(0, np.nan) * 100.0

        # 상태 기반 시그널
        long_signal = roc_val > 0  # ROC 양수 상태
        short_signal = roc_val < 0  # ROC 음수 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class McGinleyDynamicConfirmation(ConfirmationIndicator):
    """McGinley Dynamic 확인 지표.

    McGinley Dynamic 이동평균 위에 종가가 있으면 Long,
    아래에 있으면 Short 상태를 반환한다.
    MD = MD[1] + (close - MD[1]) / (length * (close / MD[1])^4)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "McGinley Dynamic Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # McGinley Dynamic 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """McGinley Dynamic 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: close > MD이면 Long 상태.
        """
        length = int(params["length"])  # McGinley Dynamic 기간
        close = df["Close"]  # 종가 시계열
        n = len(close)  # 데이터 길이
        close_vals = close.values  # 종가 numpy 배열

        # McGinley Dynamic 순차 계산
        md = np.full(n, np.nan)  # McGinley Dynamic 값 배열
        md[0] = close_vals[0]  # 초기값 = 첫 종가

        for i in range(1, n):
            if np.isnan(md[i - 1]) or md[i - 1] == 0:
                md[i] = close_vals[i]
                continue
            ratio = close_vals[i] / md[i - 1]  # close / MD[1]
            if ratio <= 0:
                md[i] = md[i - 1]
                continue
            # MD = MD[1] + (close - MD[1]) / (length * (close/MD[1])^4)
            denominator = length * (ratio ** 4)  # 분모
            if denominator == 0:
                md[i] = md[i - 1]
                continue
            md[i] = md[i - 1] + (close_vals[i] - md[i - 1]) / denominator

        md_series = pd.Series(md, index=close.index)  # McGinley Dynamic 시리즈

        # 상태 기반 시그널
        long_signal = close > md_series  # 종가가 MD 위 (상승 상태)
        short_signal = close < md_series  # 종가가 MD 아래 (하락 상태)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DPOConfirmation(ConfirmationIndicator):
    """DPO (Detrended Price Oscillator) 확인 지표.

    DPO가 0 위에 있으면 Long, 0 아래에 있으면 Short 상태를 반환한다.
    DPO = Close - SMA(Close, period).shift(period // 2 + 1)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "DPO Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 10,  # DPO 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """DPO 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: DPO > 0이면 Long 상태.
        """
        period = int(params["period"])  # DPO 기간
        close = df["Close"]  # 종가

        # DPO = Close - SMA(Close, period).shift(period // 2 + 1)
        sma_close = ma(close, period, "SMA")  # 종가 SMA
        shift_amount = period // 2 + 1  # 시프트 양
        dpo_val = close - sma_close.shift(shift_amount)  # DPO 값

        # 상태 기반 시그널
        long_signal = dpo_val > 0  # DPO 양수 상태
        short_signal = dpo_val < 0  # DPO 음수 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class TSIConfirmation(ConfirmationIndicator):
    """TSI (True Strength Index) 확인 지표.

    TSI와 시그널 라인 또는 제로 라인의 관계로 상태를 판단한다.
    서브타입:
    - Signal Cross: TSI > signal_line → Long, TSI < signal_line → Short
    - Zero line cross: TSI > 0 → Long, TSI < 0 → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "TSI Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "long": 25,  # 장기 평활 기간
            "short": 13,  # 단기 평활 기간
            "signal": 13,  # 시그널 라인 EMA 기간
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Signal Cross", "Zero line cross"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """TSI 확인 시그널을 계산한다.

        TSI = 100 * EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "Signal Cross" 또는 "Zero line cross".

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        long_len = int(params["long"])  # 장기 평활 기간
        short_len = int(params["short"])  # 단기 평활 기간
        signal_len = int(params["signal"])  # 시그널 EMA 기간
        close = df["Close"]  # 종가

        # TSI 계산: 100 * EMA(EMA(mom, long), short) / EMA(EMA(|mom|, long), short)
        momentum = close.diff()  # 모멘텀 (종가 변화량)
        double_smooth_mom = ma(ma(momentum, long_len, "EMA"), short_len, "EMA")  # 이중 평활 모멘텀
        double_smooth_abs = ma(ma(momentum.abs(), long_len, "EMA"), short_len, "EMA")  # 이중 평활 절대 모멘텀
        tsi = 100.0 * double_smooth_mom / double_smooth_abs.replace(0, np.nan)  # TSI 값

        signal_line = ma(tsi, signal_len, "EMA")  # 시그널 라인

        active_subtype = subtype or "Signal Cross"  # 기본 서브타입

        if active_subtype == "Zero line cross":
            # TSI > 0 → Long, TSI < 0 → Short (상태 기반)
            long_signal = tsi > 0
            short_signal = tsi < 0
        else:
            # Signal Cross: TSI > signal → Long, TSI < signal → Short (상태 기반)
            long_signal = tsi > signal_line
            short_signal = tsi < signal_line

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class BXtrenderConfirmation(ConfirmationIndicator):
    """B-Xtrender 확인 지표.

    단기/장기 오실레이터 값의 부호로 상태를 판단한다.
    서브타입:
    - Short term: short_val > 0 → Long, short_val < 0 → Short
    - Short and Long term: 둘 다 양수 → Long, 둘 다 음수 → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "B-Xtrender Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "short_l1": 5,  # 단기 RSI 기간
            "short_l2": 20,  # 단기 Stochastic 기간
            "short_l3": 15,  # 장기 RSI 기간
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Short term", "Short and Long term"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """B-Xtrender 확인 시그널을 계산한다.

        short_val = RSI(short_l1) - 50 기반 Stochastic 변환
        long_val = RSI(short_l3) - 50 기반 값

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "Short term" 또는 "Short and Long term".

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        short_l1 = int(params["short_l1"])  # 단기 RSI 기간
        short_l2 = int(params["short_l2"])  # 단기 Stochastic 기간
        short_l3 = int(params["short_l3"])  # 장기 RSI 기간
        close = df["Close"]  # 종가

        # 단기 값: RSI 기반 Stochastic 변환
        rsi_short = rsi_func(close, short_l1)  # 단기 RSI
        stoch_rsi = stoch_func(rsi_short, rsi_short, rsi_short, short_l2)  # RSI의 Stochastic
        short_val = ma(stoch_rsi, 3, "SMA") - 50  # SMA 평활 후 50 차감

        # 장기 값: RSI - 50
        rsi_long = rsi_func(close, short_l3)  # 장기 RSI
        long_val = ma(rsi_long, 3, "SMA") - 50  # SMA 평활 후 50 차감

        active_subtype = subtype or "Short term"  # 기본 서브타입

        if active_subtype == "Short and Long term":
            # 단기 + 장기 모두 양수 → Long, 모두 음수 → Short
            long_signal = (short_val > 0) & (long_val > 0)
            short_signal = (short_val < 0) & (long_val < 0)
        else:
            # Short term: 단기 값만 사용
            long_signal = short_val > 0
            short_signal = short_val < 0

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class BullBearPowerTrendConfirmation(ConfirmationIndicator):
    """Bull Bear Power 확인 지표.

    Bull Power와 Bear Power의 관계로 상태를 판단한다.
    서브타입:
    - Follow Trend: bull > 0 AND bear > 0 → Long, bull < 0 AND bear < 0 → Short
    - Without Trend: bull > bear → Long, bear > bull → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Bull Bear Power Trend Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 50,  # EMA 기간
            "atr": 5,  # ATR 기간 (정규화용)
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Follow Trend", "Without Trend"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Bull Bear Power 확인 시그널을 계산한다.

        Bull Power = High - EMA(Close, period)
        Bear Power = Low - EMA(Close, period)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "Follow Trend" 또는 "Without Trend".

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        period = int(params["period"])  # EMA 기간
        close = df["Close"]  # 종가
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        ema_val = ma(close, period, "EMA")  # 기준 EMA

        bull_power = high - ema_val  # Bull Power (고가 - EMA)
        bear_power = low - ema_val  # Bear Power (저가 - EMA)

        active_subtype = subtype or "Follow Trend"  # 기본 서브타입

        if active_subtype == "Without Trend":
            # 트렌드 무시: bull > bear → Long, bear > bull → Short
            long_signal = bull_power > bear_power
            short_signal = bear_power > bull_power
        else:
            # Follow Trend: 둘 다 양수 → Long, 둘 다 음수 → Short
            long_signal = (bull_power > 0) & (bear_power > 0)
            short_signal = (bull_power < 0) & (bear_power < 0)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class BBOscillatorConfirmation(ConfirmationIndicator):
    """BB Oscillator (Bollinger Band Oscillator) 확인 지표.

    볼린저 밴드 기반 오실레이터로 상태를 판단한다.
    서브타입:
    - Entering Band: osc > 0 → Long, osc < 0 → Short
    - Exiting Band: osc가 1 초과 후 1 이하 → Short exit, osc가 -1 미만 후 -1 이상 → Long exit
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "BB Oscillator Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # 볼린저 밴드 기간
            "stddev": 2.0,  # 표준편차 승수
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Entering Band", "Exiting Band"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """BB Oscillator 확인 시그널을 계산한다.

        osc = (Close - SMA) / (stddev * std) → 정규화된 위치 (-1 ~ +1 범위 근사)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "Entering Band" 또는 "Exiting Band".

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        length = int(params["length"])  # BB 기간
        stddev = float(params["stddev"])  # 표준편차 승수
        close = df["Close"]  # 종가

        # 볼린저 밴드 계산
        sma_val = ma(close, length, "SMA")  # 중심선 (SMA)
        std_val = close.rolling(window=length, min_periods=length).std()  # 표준편차
        band_width = stddev * std_val  # 밴드 폭

        # 오실레이터: 종가의 밴드 내 정규화 위치
        osc = (close - sma_val) / band_width.replace(0, np.nan)  # -1 ~ +1 범위 근사

        active_subtype = subtype or "Entering Band"  # 기본 서브타입

        if active_subtype == "Exiting Band":
            # Exiting Band: 밴드 이탈 후 복귀 감지 (상태 기반)
            prev_osc = osc.shift(1)  # 이전 오실레이터 값
            # osc가 1 초과였다가 1 이하로 → 상단 밴드 이탈 (Short 상태)
            # osc가 -1 미만이었다가 -1 이상으로 → 하단 밴드 이탈 (Long 상태)
            n = len(osc)
            long_arr = np.full(n, False)  # Long 시그널 배열
            short_arr = np.full(n, False)  # Short 시그널 배열
            osc_vals = osc.values  # 오실레이터 numpy 배열
            in_upper = False  # 상단 밴드 진입 상태
            in_lower = False  # 하단 밴드 진입 상태

            for i in range(1, n):
                if np.isnan(osc_vals[i]):
                    continue
                # 상단 밴드 진입/이탈 추적
                if osc_vals[i] > 1:
                    in_upper = True
                    in_lower = False
                elif osc_vals[i] < -1:
                    in_lower = True
                    in_upper = False

                if in_upper and osc_vals[i] <= 1:
                    short_arr[i] = True  # 상단 밴드 이탈 → Short
                    in_upper = False
                elif in_lower and osc_vals[i] >= -1:
                    long_arr[i] = True  # 하단 밴드 이탈 → Long
                    in_lower = False

            long_signal = pd.Series(long_arr, index=close.index)
            short_signal = pd.Series(short_arr, index=close.index)
        else:
            # Entering Band: osc > 0 → Long, osc < 0 → Short (상태 기반)
            long_signal = osc > 0
            short_signal = osc < 0

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class StochasticConfirmation(ConfirmationIndicator):
    """Stochastic 확인 지표.

    Stochastic %K와 %D의 관계 또는 과매수/과매도 수준으로 상태를 판단한다.
    서브타입:
    - CrossOver: %K > %D → Long, %K < %D → Short (상태 기반)
    - OB&OS levels: %K < 20 → Long (과매도), %K > 80 → Short (과매수)
    - %K above-below %D: %K > %D → Long, %K < %D → Short (상태 기반)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Stochastic Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # Stochastic 기간
            "smooth_k": 3,  # %K 평활 기간
            "smooth_d": 3,  # %D 평활 기간 (SMA of %K)
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["CrossOver", "OB&OS levels", "%K above-below %D"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Stochastic 확인 시그널을 계산한다.

        %K = SMA(Stoch(close, high, low, length), smooth_k)
        %D = SMA(%K, smooth_d)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "CrossOver", "OB&OS levels", 또는 "%K above-below %D".

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        length = int(params["length"])  # Stochastic 기간
        smooth_k = int(params["smooth_k"])  # %K 평활 기간
        smooth_d = int(params["smooth_d"])  # %D 평활 기간
        close = df["Close"]  # 종가
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        # Stochastic %K, %D 계산
        raw_k = stoch_func(close, high, low, length)  # Raw Stochastic %K
        k = ma(raw_k, smooth_k, "SMA")  # 평활된 %K
        d = ma(k, smooth_d, "SMA")  # %D (SMA of %K)

        active_subtype = subtype or "CrossOver"  # 기본 서브타입

        if active_subtype == "OB&OS levels":
            # 과매도 → Long, 과매수 → Short (상태 기반)
            long_signal = k < 20  # %K가 20 미만 (과매도 영역)
            short_signal = k > 80  # %K가 80 초과 (과매수 영역)
        else:
            # CrossOver / %K above-below %D: 상태 기반 비교
            long_signal = k > d  # %K가 %D 위 (상승 상태)
            short_signal = k < d  # %K가 %D 아래 (하락 상태)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RSIConfirmation(ConfirmationIndicator):
    """RSI 확인 지표.

    RSI와 RSI MA의 관계, 과매수/과매도 이탈, 또는 50 레벨로 상태를 판단한다.
    서브타입:
    - RSI MA Cross: RSI > RSI_MA → Long, RSI < RSI_MA → Short (상태 기반)
    - RSI Exits OB-OS: RSI가 과매도 이탈(30 상향) → Long, 과매수 이탈(70 하향) → Short
    - RSI Level: RSI > 50 → Long, RSI < 50 → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "RSI Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # RSI 기간
            "ma_length": 14,  # RSI MA 기간
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["RSI MA Cross", "RSI Exits OB-OS", "RSI Level"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """RSI 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "RSI MA Cross", "RSI Exits OB-OS", 또는 "RSI Level".

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        length = int(params["length"])  # RSI 기간
        ma_length = int(params["ma_length"])  # RSI MA 기간
        close = df["Close"]  # 종가

        rsi_val = rsi_func(close, length)  # RSI 값
        rsi_ma = ma(rsi_val, ma_length, "SMA")  # RSI의 이동평균

        active_subtype = subtype or "RSI MA Cross"  # 기본 서브타입

        if active_subtype == "RSI Exits OB-OS":
            # RSI가 과매도 이탈 → Long, 과매수 이탈 → Short (상태 기반)
            # 과매도 이탈: RSI가 30 아래에서 30 위로 올라온 후 유지
            # 과매수 이탈: RSI가 70 위에서 70 아래로 내려온 후 유지
            n = len(rsi_val)
            long_arr = np.full(n, False)  # Long 시그널 배열
            short_arr = np.full(n, False)  # Short 시그널 배열
            rsi_vals = rsi_val.values  # RSI numpy 배열
            state = 0  # 0=중립, 1=Long, -1=Short

            for i in range(1, n):
                if np.isnan(rsi_vals[i]):
                    continue
                prev = rsi_vals[i - 1]  # 이전 RSI 값
                curr = rsi_vals[i]  # 현재 RSI 값
                if not np.isnan(prev):
                    # 과매도 이탈: 이전 <= 30 이고 현재 > 30
                    if prev <= 30 and curr > 30:
                        state = 1  # Long 상태 전환
                    # 과매수 이탈: 이전 >= 70 이고 현재 < 70
                    elif prev >= 70 and curr < 70:
                        state = -1  # Short 상태 전환

                if state == 1:
                    long_arr[i] = True
                elif state == -1:
                    short_arr[i] = True

            long_signal = pd.Series(long_arr, index=close.index)
            short_signal = pd.Series(short_arr, index=close.index)
        elif active_subtype == "RSI Level":
            # RSI > 50 → Long, RSI < 50 → Short (상태 기반)
            long_signal = rsi_val > 50
            short_signal = rsi_val < 50
        else:
            # RSI MA Cross: RSI > RSI_MA → Long (상태 기반)
            long_signal = rsi_val > rsi_ma
            short_signal = rsi_val < rsi_ma

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RSIMADirectionConfirmation(ConfirmationIndicator):
    """RSI MA Direction 확인 지표.

    RSI MA의 방향(상승/하락)으로 상태를 판단한다.
    RSI MA가 상승 중이면 Long, 하락 중이면 Short.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "RSI MA Direction Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # RSI 기간
            "ma_length": 14,  # RSI MA 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """RSI MA Direction 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: RSI MA 상승 → Long, 하락 → Short 상태.
        """
        length = int(params["length"])  # RSI 기간
        ma_length = int(params["ma_length"])  # RSI MA 기간
        close = df["Close"]  # 종가

        rsi_val = rsi_func(close, length)  # RSI 값
        rsi_ma = ma(rsi_val, ma_length, "SMA")  # RSI의 이동평균

        # RSI MA 방향: 현재 > 이전 → 상승, 현재 < 이전 → 하락 (상태 기반)
        long_signal = rsi_ma > rsi_ma.shift(1)  # RSI MA 상승 중
        short_signal = rsi_ma < rsi_ma.shift(1)  # RSI MA 하락 중

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RSILimitConfirmation(ConfirmationIndicator):
    """RSI Limit 확인 지표 (필터).

    RSI가 상한/하한 범위 내에 있는지로 포지션 허용 여부를 판단한다.
    RSI < upper → Long 허용, RSI > lower → Short 허용.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "RSI Limit Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # RSI 기간
            "upper": 70,  # 상한 임계값
            "lower": 30,  # 하한 임계값
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """RSI Limit 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: RSI < upper → Long 허용, RSI > lower → Short 허용.
        """
        length = int(params["length"])  # RSI 기간
        upper = float(params["upper"])  # 상한 임계값
        lower = float(params["lower"])  # 하한 임계값
        close = df["Close"]  # 종가

        rsi_val = rsi_func(close, length)  # RSI 값

        # 필터: RSI가 과매수 아닌 경우 Long 허용, 과매도 아닌 경우 Short 허용
        long_signal = rsi_val < upper  # RSI가 상한 미만 → Long 허용
        short_signal = rsi_val > lower  # RSI가 하한 초과 → Short 허용

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RSIMALimitConfirmation(ConfirmationIndicator):
    """RSI MA Limit 확인 지표 (필터).

    RSI MA가 상한/하한 범위 내에 있는지로 포지션 허용 여부를 판단한다.
    RSI_MA < upper → Long 허용, RSI_MA > lower → Short 허용.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "RSI MA Limit Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # RSI 기간
            "ma_length": 14,  # RSI MA 기간
            "upper": 70,  # 상한 임계값
            "lower": 30,  # 하한 임계값
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """RSI MA Limit 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음 (서브타입 없음).

        Returns:
            IndicatorResult: RSI_MA < upper → Long 허용, RSI_MA > lower → Short 허용.
        """
        length = int(params["length"])  # RSI 기간
        ma_length = int(params["ma_length"])  # RSI MA 기간
        upper = float(params["upper"])  # 상한 임계값
        lower = float(params["lower"])  # 하한 임계값
        close = df["Close"]  # 종가

        rsi_val = rsi_func(close, length)  # RSI 값
        rsi_ma = ma(rsi_val, ma_length, "SMA")  # RSI의 이동평균

        # 필터: RSI MA가 과매수 아닌 경우 Long 허용, 과매도 아닌 경우 Short 허용
        long_signal = rsi_ma < upper  # RSI MA가 상한 미만 → Long 허용
        short_signal = rsi_ma > lower  # RSI MA가 하한 초과 → Short 허용

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class MACDConfirmation(ConfirmationIndicator):
    """MACD 확인 지표.

    MACD와 시그널 라인 또는 제로 라인의 관계로 상태를 판단한다.
    서브타입:
    - MACD Crossover: MACD > signal → Long, MACD < signal → Short (상태 기반)
    - Zero line crossover: MACD > 0 → Long, MACD < 0 → Short (상태 기반)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "MACD Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 12,  # 빠른 EMA 기간
            "slow": 26,  # 느린 EMA 기간
            "signal": 9,  # 시그널 라인 EMA 기간
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["MACD Crossover", "Zero line crossover"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """MACD 확인 시그널을 계산한다.

        MACD = EMA(close, fast) - EMA(close, slow)
        Signal = EMA(MACD, signal)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: "MACD Crossover" 또는 "Zero line crossover".

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        fast = int(params["fast"])  # 빠른 EMA 기간
        slow = int(params["slow"])  # 느린 EMA 기간
        signal_len = int(params["signal"])  # 시그널 EMA 기간
        close = df["Close"]  # 종가

        # MACD 계산
        fast_ema = ma(close, fast, "EMA")  # 빠른 EMA
        slow_ema = ma(close, slow, "EMA")  # 느린 EMA
        macd_line = fast_ema - slow_ema  # MACD 라인
        signal_line = ma(macd_line, signal_len, "EMA")  # 시그널 라인

        active_subtype = subtype or "MACD Crossover"  # 기본 서브타입

        if active_subtype == "Zero line crossover":
            # MACD > 0 → Long, MACD < 0 → Short (상태 기반)
            long_signal = macd_line > 0
            short_signal = macd_line < 0
        else:
            # MACD Crossover: MACD > signal → Long (상태 기반)
            long_signal = macd_line > signal_line
            short_signal = macd_line < signal_line

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


# ============================================================================
# 3차 배치: 나머지 확인 지표 (23개)
# ============================================================================


class IchimokuCloudConfirmation(ConfirmationIndicator):
    """이치모쿠 클라우드 확인 지표.

    종가가 구름(선행스팬A/B) 위에 있으면 Long, 아래에 있으면 Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Ichimoku Cloud Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "tenkan": 9,  # 전환선 기간
            "kijun": 26,  # 기준선 기간
            "senkou": 52,  # 선행스팬B 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """이치모쿠 클라우드 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: close vs cloud 기반 Long/Short 상태.
        """
        tenkan_period = int(params["tenkan"])  # 전환선 기간
        kijun_period = int(params["kijun"])  # 기준선 기간
        senkou_period = int(params["senkou"])  # 선행스팬B 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        # 전환선: (최고가 + 최저가) / 2 over tenkan 기간
        tenkan_high = high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2.0  # 전환선

        # 기준선: (최고가 + 최저가) / 2 over kijun 기간
        kijun_high = high.rolling(window=kijun_period, min_periods=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period, min_periods=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2.0  # 기준선

        # 선행스팬A: (전환선 + 기준선) / 2
        senkou_a = (tenkan_sen + kijun_sen) / 2.0  # 선행스팬A

        # 선행스팬B: (최고가 + 최저가) / 2 over senkou 기간
        senkou_b_high = high.rolling(window=senkou_period, min_periods=senkou_period).max()
        senkou_b_low = low.rolling(window=senkou_period, min_periods=senkou_period).min()
        senkou_b = (senkou_b_high + senkou_b_low) / 2.0  # 선행스팬B

        # 구름 상단/하단
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)  # 구름 상단
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)  # 구름 하단

        # close가 구름 위 → Long, 구름 아래 → Short
        long_signal = close > cloud_top  # 종가가 구름 위
        short_signal = close < cloud_bottom  # 종가가 구름 아래

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class SuperIchiConfirmation(ConfirmationIndicator):
    """SuperIchi 확인 지표.

    이치모쿠 4조건 동시 충족 시 Long/Short (상태 기반).
    Long: close > cloud_top AND tenkan > kijun AND close > kijun AND senkou_a > senkou_b
    Short: close < cloud_bottom AND tenkan < kijun AND close < kijun AND senkou_a < senkou_b
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "SuperIchi Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "tenkan": 9,  # 전환선 기간
            "kijun": 26,  # 기준선 기간
            "senkou": 52,  # 선행스팬B 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """SuperIchi 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: 4조건 동시 충족 기반 Long/Short 상태.
        """
        tenkan_period = int(params["tenkan"])  # 전환선 기간
        kijun_period = int(params["kijun"])  # 기준선 기간
        senkou_period = int(params["senkou"])  # 선행스팬B 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        # 전환선
        tenkan_sen = (
            high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
            + low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
        ) / 2.0

        # 기준선
        kijun_sen = (
            high.rolling(window=kijun_period, min_periods=kijun_period).max()
            + low.rolling(window=kijun_period, min_periods=kijun_period).min()
        ) / 2.0

        # 선행스팬A/B
        senkou_a = (tenkan_sen + kijun_sen) / 2.0  # 선행스팬A
        senkou_b = (
            high.rolling(window=senkou_period, min_periods=senkou_period).max()
            + low.rolling(window=senkou_period, min_periods=senkou_period).min()
        ) / 2.0  # 선행스팬B

        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)  # 구름 상단
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)  # 구름 하단

        # 4조건 동시 충족
        long_signal = (
            (close > cloud_top)  # 종가 > 구름 상단
            & (tenkan_sen > kijun_sen)  # 전환선 > 기준선
            & (close > kijun_sen)  # 종가 > 기준선
            & (senkou_a > senkou_b)  # 선행스팬A > 선행스팬B
        )
        short_signal = (
            (close < cloud_bottom)  # 종가 < 구름 하단
            & (tenkan_sen < kijun_sen)  # 전환선 < 기준선
            & (close < kijun_sen)  # 종가 < 기준선
            & (senkou_a < senkou_b)  # 선행스팬A < 선행스팬B
        )

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class TrendlineBreakoutConfirmation(ConfirmationIndicator):
    """추세선 돌파 확인 지표.

    피봇 기반 추세선을 계산하여 종가가 추세선 위면 Long, 아래면 Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Trendline Breakout Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # 피봇 룩백 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """추세선 돌파 확인 시그널을 계산한다.

        피봇 고점/저점을 연결한 추세선 대비 종가 위치로 판단한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: 추세선 대비 Long/Short 상태.
        """
        length = int(params["length"])  # 피봇 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        # 피봇 고점/저점을 이동 최고/최저로 근사
        upper_line = high.rolling(window=length, min_periods=length).max()  # 상단 추세선
        lower_line = low.rolling(window=length, min_periods=length).min()  # 하단 추세선

        # 종가가 상단 추세선 위 → Long, 하단 추세선 아래 → Short
        long_signal = close > upper_line  # 상단 돌파
        short_signal = close < lower_line  # 하단 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RangeDetectorConfirmation(ConfirmationIndicator):
    """레인지 감지 확인 지표.

    ATR 기반 레인지를 감지하여 상단 돌파 시 Long, 하단 돌파 시 Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Range Detector Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # 기본 기간
            "mult": 1.0,  # ATR 승수
            "atr_len": 500,  # ATR 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """레인지 감지 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: 레인지 돌파 기반 Long/Short 상태.
        """
        length = int(params["length"])  # 기본 기간
        mult = float(params["mult"])  # ATR 승수
        atr_len = int(params["atr_len"])  # ATR 기간
        close = df["Close"]  # 종가

        # ATR 기반 밴드 계산
        atr_val = atr(df, atr_len)  # ATR 값
        mid = ma(close, length, "SMA")  # 중심선 (SMA)
        upper = mid + mult * atr_val  # 상단 밴드
        lower = mid - mult * atr_val  # 하단 밴드

        # 종가가 상단 밴드 위 → Long, 하단 밴드 아래 → Short
        long_signal = close > upper  # 상단 돌파
        short_signal = close < lower  # 하단 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class HACOLTConfirmation(ConfirmationIndicator):
    """HACOLT (Heiken-Ashi Candlestick Oscillator) 확인 지표.

    HA 오실레이터가 강세면 Long, 약세면 Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "HACOLT Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "tema_period": 55,  # TEMA 기간
            "ema_period": 60,  # EMA 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """HACOLT 확인 시그널을 계산한다.

        Heiken-Ashi 종가의 TEMA와 EMA를 비교하여 방향을 판단한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: HA 오실레이터 기반 Long/Short 상태.
        """
        tema_period = int(params["tema_period"])  # TEMA 기간
        ema_period = int(params["ema_period"])  # EMA 기간
        close = df["Close"]  # 종가
        open_ = df["Open"]  # 시가
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        # Heiken-Ashi 종가 계산
        ha_close = (open_ + high + low + close) / 4.0  # HA 종가

        # TEMA 계산: 3*EMA1 - 3*EMA2 + EMA3
        ema1 = ma(ha_close, tema_period, "EMA")  # 1차 EMA
        ema2 = ma(ema1, tema_period, "EMA")  # 2차 EMA
        ema3 = ma(ema2, tema_period, "EMA")  # 3차 EMA
        tema = 3.0 * ema1 - 3.0 * ema2 + ema3  # TEMA

        # EMA 필터
        ema_filter = ma(ha_close, ema_period, "EMA")  # EMA 필터

        # TEMA > EMA → 강세(Long), TEMA < EMA → 약세(Short)
        long_signal = tema > ema_filter  # 강세 상태
        short_signal = tema < ema_filter  # 약세 상태

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ChandelierExitConfirmation(ConfirmationIndicator):
    """샹들리에 엑시트 확인 지표.

    ATR 기반 추세 추적으로 direction == 1 → Long, direction == -1 → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Chandelier Exit Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "atr_period": 22,  # ATR 기간
            "mult": 3.0,  # ATR 승수
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """샹들리에 엑시트 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: 방향 기반 Long/Short 상태.
        """
        atr_period = int(params["atr_period"])  # ATR 기간
        mult = float(params["mult"])  # ATR 승수
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        atr_val = atr(df, atr_period)  # ATR 값

        # 롱 스톱: 최고가 - mult * ATR
        highest = high.rolling(window=atr_period, min_periods=atr_period).max()  # 기간 내 최고가
        long_stop = highest - mult * atr_val  # 롱 스톱 레벨

        # 숏 스톱: 최저가 + mult * ATR
        lowest = low.rolling(window=atr_period, min_periods=atr_period).min()  # 기간 내 최저가
        short_stop = lowest + mult * atr_val  # 숏 스톱 레벨

        # 방향 결정: close > short_stop → Long(1), close < long_stop → Short(-1)
        direction = pd.Series(0, index=df.index, dtype=float)  # 방향 초기화
        for i in range(1, len(df)):
            prev_dir = direction.iloc[i - 1]  # 이전 방향
            c = close.iloc[i]  # 현재 종가

            # 이전 롱 스톱 갱신 (상승만 허용)
            if not np.isnan(long_stop.iloc[i]) and not np.isnan(long_stop.iloc[i - 1]):
                if close.iloc[i - 1] > long_stop.iloc[i - 1]:
                    long_stop.iloc[i] = max(long_stop.iloc[i], long_stop.iloc[i - 1])

            # 이전 숏 스톱 갱신 (하락만 허용)
            if not np.isnan(short_stop.iloc[i]) and not np.isnan(short_stop.iloc[i - 1]):
                if close.iloc[i - 1] < short_stop.iloc[i - 1]:
                    short_stop.iloc[i] = min(short_stop.iloc[i], short_stop.iloc[i - 1])

            if prev_dir <= 0 and c > short_stop.iloc[i]:
                direction.iloc[i] = 1  # 상승 전환
            elif prev_dir >= 0 and c < long_stop.iloc[i]:
                direction.iloc[i] = -1  # 하락 전환
            else:
                direction.iloc[i] = prev_dir  # 방향 유지

        long_signal = direction == 1  # 상승 방향
        short_signal = direction == -1  # 하락 방향

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class CCIConfirmation(ConfirmationIndicator):
    """CCI (Commodity Channel Index) 확인 지표.

    CCI > 0 → Long, CCI < 0 → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "CCI Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # CCI 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """CCI 확인 시그널을 계산한다.

        CCI = (TP - SMA(TP)) / (0.015 * MAD(TP))

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: CCI 제로라인 기반 Long/Short 상태.
        """
        length = int(params["length"])  # CCI 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        tp = (high + low + close) / 3.0  # Typical Price
        tp_sma = ma(tp, length, "SMA")  # TP의 SMA

        # 평균 절대 편차 (MAD)
        mad = tp.rolling(window=length, min_periods=length).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )

        cci = (tp - tp_sma) / (0.015 * mad.replace(0, np.nan))  # CCI 계산

        long_signal = cci > 0  # CCI 양수 → Long
        short_signal = cci < 0  # CCI 음수 → Short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ParabolicSARConfirmation(ConfirmationIndicator):
    """파라볼릭 SAR 확인 지표.

    SAR 방향이 상승(1)이면 Long, 하락(-1)이면 Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Parabolic SAR Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "start": 0.02,  # 초기 가속 계수
            "increment": 0.02,  # 가속 계수 증분
            "max": 0.2,  # 최대 가속 계수
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """파라볼릭 SAR 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: SAR 방향 기반 Long/Short 상태.
        """
        af_start = float(params["start"])  # 초기 가속 계수
        af_increment = float(params["increment"])  # 가속 계수 증분
        af_max = float(params["max"])  # 최대 가속 계수
        high = df["High"].values  # 고가 배열
        low = df["Low"].values  # 저가 배열
        close = df["Close"].values  # 종가 배열
        n = len(df)  # 데이터 길이

        direction = np.zeros(n)  # 방향 배열 (1=상승, -1=하락)
        sar = np.zeros(n)  # SAR 값 배열
        ep = np.zeros(n)  # 극점 (Extreme Point) 배열
        af = np.zeros(n)  # 가속 계수 배열

        # 초기화
        direction[0] = 1  # 초기 상승 가정
        sar[0] = low[0]  # 초기 SAR = 첫 저가
        ep[0] = high[0]  # 초기 극점 = 첫 고가
        af[0] = af_start  # 초기 가속 계수

        for i in range(1, n):
            prev_sar = sar[i - 1]  # 이전 SAR
            prev_ep = ep[i - 1]  # 이전 극점
            prev_af = af[i - 1]  # 이전 가속 계수
            prev_dir = direction[i - 1]  # 이전 방향

            # SAR 갱신
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            if prev_dir == 1:  # 상승 추세
                new_sar = min(new_sar, low[i - 1])  # SAR은 이전 저가 이하
                if i >= 2:
                    new_sar = min(new_sar, low[i - 2])

                if low[i] < new_sar:  # 반전: 하락 전환
                    direction[i] = -1
                    sar[i] = prev_ep  # SAR = 이전 극점
                    ep[i] = low[i]  # 새 극점 = 현재 저가
                    af[i] = af_start  # 가속 계수 초기화
                else:
                    direction[i] = 1  # 상승 유지
                    sar[i] = new_sar
                    if high[i] > prev_ep:
                        ep[i] = high[i]  # 극점 갱신
                        af[i] = min(prev_af + af_increment, af_max)  # 가속 계수 증가
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:  # 하락 추세
                new_sar = max(new_sar, high[i - 1])  # SAR은 이전 고가 이상
                if i >= 2:
                    new_sar = max(new_sar, high[i - 2])

                if high[i] > new_sar:  # 반전: 상승 전환
                    direction[i] = 1
                    sar[i] = prev_ep  # SAR = 이전 극점
                    ep[i] = high[i]  # 새 극점 = 현재 고가
                    af[i] = af_start  # 가속 계수 초기화
                else:
                    direction[i] = -1  # 하락 유지
                    sar[i] = new_sar
                    if low[i] < prev_ep:
                        ep[i] = low[i]  # 극점 갱신
                        af[i] = min(prev_af + af_increment, af_max)  # 가속 계수 증가
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

        dir_series = pd.Series(direction, index=df.index)  # 방향 시리즈
        long_signal = dir_series == 1  # 상승 방향
        short_signal = dir_series == -1  # 하락 방향

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class SSLChannelConfirmation(ConfirmationIndicator):
    """SSL 채널 확인 지표.

    ssl_up > ssl_down → Long, ssl_up < ssl_down → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "SSL Channel Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 10,  # SSL 채널 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """SSL 채널 확인 시그널을 계산한다.

        ssl_up = SMA(High, period), ssl_down = SMA(Low, period)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: SSL 채널 기반 Long/Short 상태.
        """
        period = int(params["period"])  # SSL 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        sma_high = ma(high, period, "SMA")  # 고가 SMA
        sma_low = ma(low, period, "SMA")  # 저가 SMA

        # 방향 결정: close > sma_high → 상승, close < sma_low → 하락
        hlv = pd.Series(np.nan, index=df.index)  # High-Low Value
        hlv[close > sma_high] = 1  # 상승
        hlv[close < sma_low] = -1  # 하락
        hlv = hlv.ffill().fillna(1)  # 전방 채움

        ssl_down = np.where(hlv < 0, sma_high, sma_low)  # SSL Down
        ssl_up = np.where(hlv < 0, sma_low, sma_high)  # SSL Up

        ssl_up_s = pd.Series(ssl_up, index=df.index)  # SSL Up 시리즈
        ssl_down_s = pd.Series(ssl_down, index=df.index)  # SSL Down 시리즈

        long_signal = ssl_up_s > ssl_down_s  # SSL Up > SSL Down → Long
        short_signal = ssl_up_s < ssl_down_s  # SSL Up < SSL Down → Short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class HullSuiteConfirmation(ConfirmationIndicator):
    """Hull Suite 확인 지표.

    Hull MA가 상승 중이면 Long, 하락 중이면 Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Hull Suite Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 55,  # Hull MA 기간
            "mode": "Hma",  # MA 모드 (Hma/Ehma/Thma)
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Hull Suite 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: Hull MA 방향 기반 Long/Short 상태.
        """
        length = int(params["length"])  # Hull 기간
        close = df["Close"]  # 종가

        # HMA 계산
        hull = ma(close, length, "HMA")  # Hull Moving Average

        # Hull MA 방향: 현재 > 이전 → 상승, 현재 < 이전 → 하락
        long_signal = hull > hull.shift(1)  # 상승 중
        short_signal = hull < hull.shift(1)  # 하락 중

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class AwesomeOscillatorConfirmation(ConfirmationIndicator):
    """Awesome Oscillator 확인 지표.

    서브타입:
    - Zero Line Cross: AO > 0 → Long, AO < 0 → Short
    - AC Zero Line Cross: AC > 0 → Long, AC < 0 → Short
    - AC Momentum Bar: AC 증가 → Long, AC 감소 → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Awesome Oscillator Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 5,  # 빠른 SMA 기간
            "slow": 34,  # 느린 SMA 기간
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Zero Line Cross", "AC Zero Line Cross", "AC Momentum Bar"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Awesome Oscillator 확인 시그널을 계산한다.

        AO = SMA(HL2, fast) - SMA(HL2, slow)
        AC = AO - SMA(AO, 5)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 서브타입 이름.

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        fast = int(params["fast"])  # 빠른 기간
        slow = int(params["slow"])  # 느린 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        hl2 = (high + low) / 2.0  # HL2 (중간가)
        ao = ma(hl2, fast, "SMA") - ma(hl2, slow, "SMA")  # Awesome Oscillator
        ac = ao - ma(ao, 5, "SMA")  # Accelerator Oscillator

        active_subtype = subtype or "Zero Line Cross"  # 기본 서브타입

        if active_subtype == "AC Zero Line Cross":
            # AC > 0 → Long, AC < 0 → Short
            long_signal = ac > 0
            short_signal = ac < 0
        elif active_subtype == "AC Momentum Bar":
            # AC 증가 → Long, AC 감소 → Short
            long_signal = ac > ac.shift(1)
            short_signal = ac < ac.shift(1)
        else:
            # Zero Line Cross: AO > 0 → Long, AO < 0 → Short
            long_signal = ao > 0
            short_signal = ao < 0

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DMIADXConfirmation(ConfirmationIndicator):
    """DMI-ADX 확인 지표.

    서브타입:
    - Adx Only: ADX > 20 → Long AND Short (추세 강도 필터)
    - Adx & +Di -Di: +DI > -DI AND ADX > 20 → Long, -DI > +DI AND ADX > 20 → Short
    - Advance: +DI > -DI AND ADX > ADX[1] → Long, -DI > +DI AND ADX > ADX[1] → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "DMI ADX Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # DI 기간
            "adx_smoothing": 14,  # ADX 평활 기간
            "adx_threshold": 20,  # ADX 임계값
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Adx Only", "Adx & +Di -Di", "Advance"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """DMI-ADX 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 서브타입 이름.

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        length = int(params["length"])  # DI 기간
        adx_smoothing = int(params["adx_smoothing"])  # ADX 평활 기간
        adx_threshold = float(params["adx_threshold"])  # ADX 임계값
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        # +DM, -DM 계산
        up_move = high.diff()  # 고가 변화
        down_move = -low.diff()  # 저가 변화 (부호 반전)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)  # +DM
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)  # -DM

        plus_dm_s = pd.Series(plus_dm, index=df.index)  # +DM 시리즈
        minus_dm_s = pd.Series(minus_dm, index=df.index)  # -DM 시리즈

        atr_val = atr(df, length)  # ATR 값

        # +DI, -DI 계산
        plus_di = 100.0 * ma(plus_dm_s, length, "RMA") / atr_val.replace(0, np.nan)  # +DI
        minus_di = 100.0 * ma(minus_dm_s, length, "RMA") / atr_val.replace(0, np.nan)  # -DI

        # DX, ADX 계산
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)  # DX
        adx = ma(dx, adx_smoothing, "RMA")  # ADX

        active_subtype = subtype or "Adx & +Di -Di"  # 기본 서브타입

        if active_subtype == "Adx Only":
            # ADX > threshold → 추세 존재 (Long AND Short 모두 허용)
            trend_filter = adx > adx_threshold  # 추세 필터
            long_signal = trend_filter
            short_signal = trend_filter
        elif active_subtype == "Advance":
            # +DI > -DI AND ADX 상승 → Long
            long_signal = (plus_di > minus_di) & (adx > adx.shift(1))
            short_signal = (minus_di > plus_di) & (adx > adx.shift(1))
        else:
            # Adx & +Di -Di: +DI > -DI AND ADX > threshold → Long
            long_signal = (plus_di > minus_di) & (adx > adx_threshold)
            short_signal = (minus_di > plus_di) & (adx > adx_threshold)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class WaddahAttarExplosionConfirmation(ConfirmationIndicator):
    """Waddah Attar Explosion 확인 지표.

    trend > 0 AND trend > explosion → Long,
    trend < 0 AND |trend| > explosion → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Waddah Attar Explosion Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "sensitivity": 150,  # 감도
            "fast": 20,  # 빠른 EMA 기간
            "slow": 40,  # 느린 EMA 기간
            "bb_length": 20,  # 볼린저 밴드 기간
            "bb_mult": 2.0,  # 볼린저 밴드 승수
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Waddah Attar Explosion 확인 시그널을 계산한다.

        trend = MACD 차이 * sensitivity
        explosion = BB 상단 - BB 하단

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: trend vs explosion 기반 Long/Short 상태.
        """
        sensitivity = float(params["sensitivity"])  # 감도
        fast = int(params["fast"])  # 빠른 EMA 기간
        slow = int(params["slow"])  # 느린 EMA 기간
        bb_length = int(params["bb_length"])  # BB 기간
        bb_mult = float(params["bb_mult"])  # BB 승수
        close = df["Close"]  # 종가

        # MACD 기반 트렌드
        macd1 = ma(close, fast, "EMA") - ma(close, slow, "EMA")  # 현재 MACD
        macd2 = ma(close.shift(1), fast, "EMA") - ma(close.shift(1), slow, "EMA")  # 이전 MACD
        trend = (macd1 - macd2) * sensitivity  # 트렌드 값

        # 볼린저 밴드 기반 폭발선
        bb_sma = ma(close, bb_length, "SMA")  # BB 중심선
        bb_std = close.rolling(window=bb_length, min_periods=bb_length).std()  # BB 표준편차
        bb_upper = bb_sma + bb_mult * bb_std  # BB 상단
        bb_lower = bb_sma - bb_mult * bb_std  # BB 하단
        explosion = bb_upper - bb_lower  # 폭발선

        # trend > 0 AND trend > explosion → Long
        long_signal = (trend > 0) & (trend > explosion)
        # trend < 0 AND |trend| > explosion → Short
        short_signal = (trend < 0) & (trend.abs() > explosion)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class VolatilityOscillatorConfirmation(ConfirmationIndicator):
    """변동성 오실레이터 확인 지표.

    spike > upper → Long, spike < lower → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Volatility Oscillator Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 100,  # 변동성 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """변동성 오실레이터 확인 시그널을 계산한다.

        spike = close - open
        upper/lower = ±stdev(spike, length)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: spike vs 밴드 기반 Long/Short 상태.
        """
        length = int(params["length"])  # 변동성 기간
        close = df["Close"]  # 종가
        open_ = df["Open"]  # 시가

        spike = close - open_  # 스파이크 (종가 - 시가)
        std = spike.rolling(window=length, min_periods=length).std()  # 표준편차
        upper = std  # 상단 밴드
        lower = -std  # 하단 밴드

        long_signal = spike > upper  # 스파이크 > 상단 → Long
        short_signal = spike < lower  # 스파이크 < 하단 → Short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ChoppinessIndexConfirmation(ConfirmationIndicator):
    """Choppiness Index 확인 지표.

    CI < 38.2 → 추세 존재 (Long+Short 허용), CI > 61.8 → 횡보 (둘 다 불허).
    확인 전용 필터 지표.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Choppiness Index Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # CI 기간
            "trending_threshold": 38.2,  # 추세 임계값
            "choppy_threshold": 61.8,  # 횡보 임계값
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Choppiness Index 확인 시그널을 계산한다.

        CI = 100 * LOG10(SUM(ATR,length) / (HH - LL)) / LOG10(length)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: CI 기반 추세/횡보 필터.
        """
        length = int(params["length"])  # CI 기간
        trending = float(params["trending_threshold"])  # 추세 임계값
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        atr_val = atr(df, 1)  # ATR(1) = True Range
        atr_sum = atr_val.rolling(window=length, min_periods=length).sum()  # ATR 합계

        hh = high.rolling(window=length, min_periods=length).max()  # 기간 내 최고가
        ll = low.rolling(window=length, min_periods=length).min()  # 기간 내 최저가
        hl_range = (hh - ll).replace(0, np.nan)  # 고저 범위

        ci = 100.0 * np.log10(atr_sum / hl_range) / np.log10(length)  # CI 계산

        # CI < trending → 추세 존재 (Long AND Short 모두 허용)
        trending_filter = ci < trending  # 추세 필터
        long_signal = trending_filter
        short_signal = trending_filter

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DamianiVolatilityConfirmation(ConfirmationIndicator):
    """Damiani Volatility 확인 지표.

    서브타입:
    - Simple: vol > anti_vol → Long AND Short (변동성 필터)
    - Threshold: vol/anti_vol > threshold → 필터 통과
    - 10p Difference: (vol - anti_vol)/anti_vol > 0.1 → 필터 통과
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Damiani Volatility Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "atr_period": 13,  # ATR 기간
            "anti_atr_period": 40,  # Anti-ATR 기간
            "threshold": 1.3,  # 임계값 (Threshold 서브타입용)
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Simple", "Threshold", "10p Difference"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Damiani Volatility 확인 시그널을 계산한다.

        vol = ATR(atr_period), anti_vol = ATR(anti_atr_period)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 서브타입 이름.

        Returns:
            IndicatorResult: 변동성 필터 기반 Long/Short 상태.
        """
        atr_period = int(params["atr_period"])  # ATR 기간
        anti_atr_period = int(params["anti_atr_period"])  # Anti-ATR 기간
        threshold = float(params["threshold"])  # 임계값

        vol = atr(df, atr_period)  # 변동성 (짧은 ATR)
        anti_vol = atr(df, anti_atr_period)  # 반변동성 (긴 ATR)

        active_subtype = subtype or "Simple"  # 기본 서브타입

        if active_subtype == "Threshold":
            # vol/anti_vol > threshold → 필터 통과
            ratio = vol / anti_vol.replace(0, np.nan)  # 비율
            vol_filter = ratio > threshold
        elif active_subtype == "10p Difference":
            # (vol - anti_vol)/anti_vol > 0.1 → 필터 통과
            diff_ratio = (vol - anti_vol) / anti_vol.replace(0, np.nan)  # 차이 비율
            vol_filter = diff_ratio > 0.1
        else:
            # Simple: vol > anti_vol → 필터 통과
            vol_filter = vol > anti_vol

        # 변동성 필터 (Long AND Short 모두 동일)
        long_signal = vol_filter
        short_signal = vol_filter

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class VolumeConfirmation(ConfirmationIndicator):
    """거래량 확인 지표.

    서브타입:
    - volume above MA: volume > SMA(volume, length) → Long AND Short (거래량 필터)
    - Simple: close > open AND volume > MA → Long, close < open AND volume > MA → Short
    - Delta: 누적 델타 > 0 → Long, < 0 → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Volume Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # 거래량 MA 기간
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["volume above MA", "Simple", "Delta"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """거래량 확인 시그널을 계산한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 서브타입 이름.

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        length = int(params["length"])  # MA 기간
        close = df["Close"]  # 종가
        open_ = df["Open"]  # 시가
        volume = df["Volume"].astype(float)  # 거래량

        vol_ma = ma(volume, length, "SMA")  # 거래량 이동평균
        vol_above = volume > vol_ma  # 거래량 > MA

        active_subtype = subtype or "volume above MA"  # 기본 서브타입

        if active_subtype == "Simple":
            # close > open AND volume > MA → Long
            long_signal = (close > open_) & vol_above
            short_signal = (close < open_) & vol_above
        elif active_subtype == "Delta":
            # 누적 델타: 양봉 → 매수량, 음봉 → 매도량
            buy_vol = np.where(close > open_, volume, 0.0)  # 매수 거래량
            sell_vol = np.where(close < open_, volume, 0.0)  # 매도 거래량
            delta = pd.Series(buy_vol - sell_vol, index=df.index)  # 델타
            cum_delta = delta.rolling(window=length, min_periods=1).sum()  # 누적 델타

            long_signal = cum_delta > 0  # 누적 델타 양수 → Long
            short_signal = cum_delta < 0  # 누적 델타 음수 → Short
        else:
            # volume above MA: 거래량 필터 (Long AND Short 동일)
            long_signal = vol_above
            short_signal = vol_above

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class WolfpackIdConfirmation(ConfirmationIndicator):
    """Wolfpack Id 확인 지표.

    빠른/느린 EMA 스프레드 > 0 → Long, < 0 → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Wolfpack Id Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 3,  # 빠른 EMA 기간
            "slow": 8,  # 느린 EMA 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Wolfpack Id 확인 시그널을 계산한다.

        spread = EMA(close, fast) - EMA(close, slow)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: 스프레드 기반 Long/Short 상태.
        """
        fast = int(params["fast"])  # 빠른 EMA 기간
        slow = int(params["slow"])  # 느린 EMA 기간
        close = df["Close"]  # 종가

        fast_ema = ma(close, fast, "EMA")  # 빠른 EMA
        slow_ema = ma(close, slow, "EMA")  # 느린 EMA
        spread = fast_ema - slow_ema  # 스프레드

        long_signal = spread > 0  # 스프레드 양수 → Long
        short_signal = spread < 0  # 스프레드 음수 → Short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class QQEModConfirmation(ConfirmationIndicator):
    """QQE Mod 확인 지표.

    서브타입:
    - Line: RSI smoothed > QQE line → Long, < → Short
    - Bar: histogram > 0 → Long, < 0 → Short
    - Line & Bar: 두 조건 모두 충족 → Long/Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "QQE Mod Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "rsi_period": 6,  # RSI 기간
            "sf": 5,  # RSI 평활 기간
            "qqe_factor": 3.0,  # QQE 팩터
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Line", "Bar", "Line & Bar"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """QQE Mod 확인 시그널을 계산한다.

        RSI smoothed와 QQE 라인의 관계로 방향을 판단한다.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 서브타입 이름.

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        rsi_period = int(params["rsi_period"])  # RSI 기간
        sf = int(params["sf"])  # 평활 기간
        qqe_factor = float(params["qqe_factor"])  # QQE 팩터
        close = df["Close"]  # 종가

        # RSI 계산 및 평활화
        rsi_val = rsi_func(close, rsi_period)  # RSI
        rsi_smooth = ma(rsi_val, sf, "EMA")  # RSI 평활화

        # QQE 라인 계산
        rsi_diff = rsi_smooth.diff().abs()  # RSI 변화 절대값
        wilders_period = rsi_period * 2 - 1  # Wilder's 기간
        atr_rsi = ma(rsi_diff, wilders_period, "EMA")  # RSI의 ATR
        dar = ma(atr_rsi, wilders_period, "EMA") * qqe_factor  # Dynamic Average Range

        # QQE 밴드 (trailing stop 방식)
        long_band = pd.Series(0.0, index=df.index)  # 롱 밴드
        short_band = pd.Series(0.0, index=df.index)  # 숏 밴드
        trend = pd.Series(0, index=df.index)  # 추세 방향

        for i in range(1, len(df)):
            new_long = rsi_smooth.iloc[i] - dar.iloc[i]  # 새 롱 밴드
            new_short = rsi_smooth.iloc[i] + dar.iloc[i]  # 새 숏 밴드

            # 롱 밴드 갱신 (상승만 허용)
            if rsi_smooth.iloc[i - 1] > long_band.iloc[i - 1] and not np.isnan(new_long):
                long_band.iloc[i] = max(new_long, long_band.iloc[i - 1])
            else:
                long_band.iloc[i] = new_long if not np.isnan(new_long) else 0.0

            # 숏 밴드 갱신 (하락만 허용)
            if rsi_smooth.iloc[i - 1] < short_band.iloc[i - 1] and not np.isnan(new_short):
                short_band.iloc[i] = min(new_short, short_band.iloc[i - 1])
            else:
                short_band.iloc[i] = new_short if not np.isnan(new_short) else 0.0

            # 추세 결정
            if rsi_smooth.iloc[i] > short_band.iloc[i - 1]:
                trend.iloc[i] = 1  # 상승 추세
            elif rsi_smooth.iloc[i] < long_band.iloc[i - 1]:
                trend.iloc[i] = -1  # 하락 추세
            else:
                trend.iloc[i] = trend.iloc[i - 1]  # 추세 유지

        # QQE 라인 = 추세에 따라 롱/숏 밴드 선택
        qqe_line = np.where(trend == 1, long_band, short_band)
        qqe_line_s = pd.Series(qqe_line, index=df.index)  # QQE 라인 시리즈

        # 히스토그램 = RSI smoothed - 50 (중심선 기준)
        histogram = rsi_smooth - 50.0  # 히스토그램

        # Line 조건
        line_long = rsi_smooth > qqe_line_s  # RSI > QQE → Long
        line_short = rsi_smooth < qqe_line_s  # RSI < QQE → Short

        # Bar 조건
        bar_long = histogram > 0  # 히스토그램 양수 → Long
        bar_short = histogram < 0  # 히스토그램 음수 → Short

        active_subtype = subtype or "Line"  # 기본 서브타입

        if active_subtype == "Bar":
            long_signal = bar_long
            short_signal = bar_short
        elif active_subtype == "Line & Bar":
            long_signal = line_long & bar_long  # 두 조건 모두 충족
            short_signal = line_short & bar_short
        else:
            # Line
            long_signal = line_long
            short_signal = line_short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ChaikinMoneyFlowConfirmation(ConfirmationIndicator):
    """Chaikin Money Flow 확인 지표.

    CMF > 0 → Long, CMF < 0 → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Chaikin Money Flow Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # CMF 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Chaikin Money Flow 확인 시그널을 계산한다.

        CMF = SUM(MFV, length) / SUM(volume, length)
        MFV = ((close - low) - (high - close)) / (high - low) * volume

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: CMF 기반 Long/Short 상태.
        """
        length = int(params["length"])  # CMF 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가
        volume = df["Volume"].astype(float)  # 거래량

        # Money Flow Multiplier
        hl_range = (high - low).replace(0, np.nan)  # 고저 범위
        mfm = ((close - low) - (high - close)) / hl_range  # MF 승수

        # Money Flow Volume
        mfv = mfm * volume  # MF 거래량

        # CMF = SUM(MFV) / SUM(Volume)
        mfv_sum = mfv.rolling(window=length, min_periods=length).sum()  # MFV 합계
        vol_sum = volume.rolling(window=length, min_periods=length).sum()  # 거래량 합계
        cmf = mfv_sum / vol_sum.replace(0, np.nan)  # CMF

        long_signal = cmf > 0  # CMF 양수 → Long
        short_signal = cmf < 0  # CMF 음수 → Short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class VortexIndicatorConfirmation(ConfirmationIndicator):
    """Vortex Indicator 확인 지표.

    서브타입:
    - Simple: VI+ > VI- → Long, VI- > VI+ → Short
    - Advance: VI+ > VI- AND VI+ > 1 → Long, VI- > VI+ AND VI- > 1 → Short
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Vortex Indicator Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 14,  # Vortex 기간
        }

    @property
    def subtypes(self) -> list[str]:
        """지원하는 서브타입 목록."""
        return ["Simple", "Advance"]

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Vortex Indicator 확인 시그널을 계산한다.

        VI+ = SUM(|High - Low[1]|, period) / SUM(TR, period)
        VI- = SUM(|Low - High[1]|, period) / SUM(TR, period)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 서브타입 이름.

        Returns:
            IndicatorResult: 서브타입에 따른 Long/Short 상태.
        """
        period = int(params["period"])  # Vortex 기간
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        from sp500_backtest.indicators.utils import true_range as tr_func

        tr = tr_func(df)  # True Range

        # VM+ = |High - Low[1]|, VM- = |Low - High[1]|
        vm_plus = (high - low.shift(1)).abs()  # VM+
        vm_minus = (low - high.shift(1)).abs()  # VM-

        # SUM over period
        tr_sum = tr.rolling(window=period, min_periods=period).sum()  # TR 합계
        vm_plus_sum = vm_plus.rolling(window=period, min_periods=period).sum()  # VM+ 합계
        vm_minus_sum = vm_minus.rolling(window=period, min_periods=period).sum()  # VM- 합계

        # VI+, VI-
        vi_plus = vm_plus_sum / tr_sum.replace(0, np.nan)  # VI+
        vi_minus = vm_minus_sum / tr_sum.replace(0, np.nan)  # VI-

        active_subtype = subtype or "Simple"  # 기본 서브타입

        if active_subtype == "Advance":
            # VI+ > VI- AND VI+ > 1 → Long
            long_signal = (vi_plus > vi_minus) & (vi_plus > 1)
            short_signal = (vi_minus > vi_plus) & (vi_minus > 1)
        else:
            # Simple: VI+ > VI- → Long
            long_signal = vi_plus > vi_minus
            short_signal = vi_minus > vi_plus

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class STCConfirmation(ConfirmationIndicator):
    """Schaff Trend Cycle 확인 지표.

    STC > 50 → Long, STC < 50 → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "STC Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 23,  # 빠른 EMA 기간
            "slow": 50,  # 느린 EMA 기간
            "cycle": 10,  # Stochastic 사이클 기간
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """Schaff Trend Cycle 확인 시그널을 계산한다.

        MACD → Stochastic → EMA → Stochastic → EMA 이중 평활.

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: STC 기반 Long/Short 상태.
        """
        fast = int(params["fast"])  # 빠른 EMA 기간
        slow = int(params["slow"])  # 느린 EMA 기간
        cycle = int(params["cycle"])  # 사이클 기간
        close = df["Close"]  # 종가

        # MACD 계산
        macd_line = ma(close, fast, "EMA") - ma(close, slow, "EMA")  # MACD

        # 1차 Stochastic
        lowest1 = macd_line.rolling(window=cycle, min_periods=cycle).min()  # 최저값
        highest1 = macd_line.rolling(window=cycle, min_periods=cycle).max()  # 최고값
        range1 = (highest1 - lowest1).replace(0, np.nan)  # 범위
        stoch1 = 100.0 * (macd_line - lowest1) / range1  # 1차 Stochastic

        # 1차 EMA 평활
        pf = ma(stoch1, cycle, "EMA")  # PF (Percent Fast)

        # 2차 Stochastic
        lowest2 = pf.rolling(window=cycle, min_periods=cycle).min()  # 최저값
        highest2 = pf.rolling(window=cycle, min_periods=cycle).max()  # 최고값
        range2 = (highest2 - lowest2).replace(0, np.nan)  # 범위
        stoch2 = 100.0 * (pf - lowest2) / range2  # 2차 Stochastic

        # 2차 EMA 평활 → STC
        stc = ma(stoch2, cycle, "EMA")  # STC 값

        long_signal = stc > 50  # STC > 50 → Long
        short_signal = stc < 50  # STC < 50 → Short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class VWAPConfirmation(ConfirmationIndicator):
    """VWAP 확인 지표.

    close > VWAP → Long, close < VWAP → Short (상태 기반).
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "VWAP Confirmation"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "anchor": "Session",  # 앵커 기간 (일봉에서는 누적 VWAP 사용)
        }

    def _calculate_impl(
        self,
        df: pd.DataFrame,
        params: dict,
        subtype: str | None,
    ) -> IndicatorResult:
        """VWAP 확인 시그널을 계산한다.

        VWAP = cumsum(typical_price * volume) / cumsum(volume)

        Args:
            df: OHLCV DataFrame.
            params: 병합된 최종 파라미터.
            subtype: 사용하지 않음.

        Returns:
            IndicatorResult: close vs VWAP 기반 Long/Short 상태.
        """
        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가
        volume = df["Volume"].astype(float)  # 거래량

        tp = (high + low + close) / 3.0  # Typical Price

        # 누적 VWAP 계산
        cum_tp_vol = (tp * volume).cumsum()  # 누적 (TP × Volume)
        cum_vol = volume.cumsum()  # 누적 Volume
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)  # VWAP

        long_signal = close > vwap  # 종가 > VWAP → Long
        short_signal = close < vwap  # 종가 < VWAP → Short

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )
