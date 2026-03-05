"""리딩 지표 모듈 (Leading Indicators).

Pine Script v6 기반 37개 리딩 지표를 Python으로 재구현한다.
모든 지표는 BaseIndicator를 상속하고, pandas/numpy 벡터화 연산을 사용한다.
"""

import numpy as np
import pandas as pd

from sp500_backtest.indicators.base import BaseIndicator, IndicatorResult
from sp500_backtest.indicators.utils import (
    atr,
    crossover,
    crossunder,
    ma,
    rsi as rsi_func,
    stoch,
    true_range,
)


class RangeFilter(BaseIndicator):
    """Range Filter 지표.

    smoothrng + rngfilt 함수를 사용하여 가격 필터링 후 방향 변화를 감지한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Range Filter"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 100,  # 평활 기간
            "mult": 3.0,  # 범위 승수
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Range Filter 시그널을 계산한다."""
        p = self._resolve_params(params)
        period = int(p["period"])  # 평활 기간
        mult = float(p["mult"])  # 범위 승수
        close = df["Close"]  # 종가 시계열
        n = len(close)  # 데이터 길이

        # smoothrng: EMA(|close - close[1]|, period) * mult
        abs_diff = close.diff().abs()  # |종가 변화량|
        smooth_rng = abs_diff.ewm(span=period, adjust=False).mean() * mult  # 평활 범위

        # rngfilt: 반복 계산 (벡터화 불가, 순차 처리)
        filt = np.full(n, np.nan)  # 필터 값 배열
        direction = np.zeros(n, dtype=int)  # 방향 추적 (1=상승, -1=하락)

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

        # 방향 추적
        filt_shifted = filt_series.shift(1)  # 이전 필터 값
        upward = (filt_series > filt_shifted).astype(int)  # 상승 여부
        downward = (filt_series < filt_shifted).astype(int)  # 하락 여부

        # 방향 변화 시 시그널 생성
        prev_upward = upward.shift(1).fillna(0).astype(int)  # 이전 상승 여부
        prev_downward = downward.shift(1).fillna(0).astype(int)  # 이전 하락 여부

        long_signal = (upward == 1) & (prev_upward == 0)  # 상승 전환
        short_signal = (downward == 1) & (prev_downward == 0)  # 하락 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RQK(BaseIndicator):
    """Rational Quadratic Kernel (RQK) 지표.

    Nadaraya-Watson 커널 회귀를 사용하여 트렌드 방향을 감지한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "RQK"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "lookback": 8,  # 룩백 기간
            "relative_weight": 8,  # 상대 가중치 (alpha)
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """RQK 시그널을 계산한다."""
        p = self._resolve_params(params)
        lookback = int(p["lookback"])  # 룩백 기간
        alpha = float(p["relative_weight"])  # 상대 가중치
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

        # 방향 변화 감지
        rising = yhat_series > yhat_prev  # 상승 중
        falling = yhat_series < yhat_prev  # 하락 중
        prev_rising = rising.shift(1).fillna(False)  # 이전 상승 여부
        prev_falling = falling.shift(1).fillna(False)  # 이전 하락 여부

        long_signal = rising & ~prev_rising  # 상승 전환
        short_signal = falling & ~prev_falling  # 하락 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class Supertrend(BaseIndicator):
    """Supertrend 지표.

    ATR 기반 상/하 밴드를 사용하여 트렌드 방향을 추적한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Supertrend"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "atr_period": 10,  # ATR 기간
            "factor": 3.0,  # ATR 승수
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Supertrend 시그널을 계산한다."""
        p = self._resolve_params(params)
        atr_period = int(p["atr_period"])  # ATR 기간
        factor = float(p["factor"])  # ATR 승수

        close = df["Close"].values  # 종가 numpy 배열
        high = df["High"].values  # 고가 numpy 배열
        low = df["Low"].values  # 저가 numpy 배열
        hl2 = (high + low) / 2.0  # 중간가 (High+Low)/2
        n = len(close)  # 데이터 길이

        atr_vals = atr(df, atr_period).values  # ATR 값 배열

        upper_band = hl2 + factor * atr_vals  # 상단 밴드
        lower_band = hl2 - factor * atr_vals  # 하단 밴드

        supertrend = np.full(n, np.nan)  # Supertrend 값
        direction = np.ones(n, dtype=int)  # 방향 (1=상승, -1=하락)

        # 최종 상/하 밴드 (이전 값과 비교하여 조정)
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
        prev_dir = dir_series.shift(1)  # 이전 방향

        long_signal = (dir_series == -1) & (prev_dir == 1)  # 하락→상승 전환 (Long)
        short_signal = (dir_series == 1) & (prev_dir == -1)  # 상승→하락 전환 (Short)

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class HalfTrend(BaseIndicator):
    """Half Trend 지표.

    ATR 기반 트렌드 추적으로 방향 전환을 감지한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Half Trend"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "amplitude": 2,  # 진폭 (ATR 기간)
            "channel_deviation": 2,  # 채널 편차 승수
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Half Trend 시그널을 계산한다."""
        p = self._resolve_params(params)
        amplitude = int(p["amplitude"])  # 진폭
        channel_dev = float(p["channel_deviation"])  # 채널 편차

        close = df["Close"].values  # 종가
        high = df["High"].values  # 고가
        low = df["Low"].values  # 저가
        n = len(close)  # 데이터 길이

        atr_vals = atr(df, max(amplitude, 1)).values  # ATR 값
        dev = channel_dev * atr_vals  # 편차

        # 고가/저가의 amplitude 기간 최고/최저
        high_ma = pd.Series(high).rolling(window=max(amplitude, 1), min_periods=1).max().values  # 기간 최고가
        low_ma = pd.Series(low).rolling(window=max(amplitude, 1), min_periods=1).min().values  # 기간 최저가

        trend = np.zeros(n, dtype=int)  # 트렌드 방향 (0=상승, 1=하락)
        half_trend = np.full(n, np.nan)  # Half Trend 값

        # 초기화
        half_trend[0] = close[0]  # 초기값

        for i in range(1, n):
            if np.isnan(atr_vals[i]):
                half_trend[i] = half_trend[i - 1] if not np.isnan(half_trend[i - 1]) else close[i]
                trend[i] = trend[i - 1]
                continue

            prev_trend = trend[i - 1]  # 이전 트렌드
            prev_ht = half_trend[i - 1]  # 이전 Half Trend 값

            if prev_trend == 0:  # 상승 트렌드
                max_low = low_ma[i]  # 기간 최저가
                new_ht = max(prev_ht, max_low)  # 상승 시 최저가 추적
                if close[i] < new_ht - dev[i]:
                    trend[i] = 1  # 하락 전환
                    half_trend[i] = high_ma[i]  # 최고가로 전환
                else:
                    trend[i] = 0
                    half_trend[i] = new_ht
            else:  # 하락 트렌드
                min_high = high_ma[i]  # 기간 최고가
                new_ht = min(prev_ht, min_high)  # 하락 시 최고가 추적
                if close[i] > new_ht + dev[i]:
                    trend[i] = 0  # 상승 전환
                    half_trend[i] = low_ma[i]  # 최저가로 전환
                else:
                    trend[i] = 1
                    half_trend[i] = new_ht

        trend_series = pd.Series(trend, index=df.index)  # 트렌드 시리즈
        prev_trend_s = trend_series.shift(1)  # 이전 트렌드

        long_signal = (trend_series == 0) & (prev_trend_s == 1)  # 하락→상승 전환
        short_signal = (trend_series == 1) & (prev_trend_s == 0)  # 상승→하락 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class IchimokuCloud(BaseIndicator):
    """Ichimoku Cloud 지표.

    전환선/기준선/선행스팬을 사용하여 클라우드 돌파 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Ichimoku Cloud"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "tenkan": 9,  # 전환선 기간
            "kijun": 26,  # 기준선 기간
            "senkou": 52,  # 선행스팬B 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Ichimoku Cloud 시그널을 계산한다."""
        p = self._resolve_params(params)
        tenkan_period = int(p["tenkan"])  # 전환선 기간
        kijun_period = int(p["kijun"])  # 기준선 기간
        senkou_period = int(p["senkou"])  # 선행스팬B 기간

        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        # 전환선 (Tenkan-sen): (최고가 + 최저가) / 2 over tenkan 기간
        tenkan = (
            high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
            + low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
        ) / 2.0

        # 기준선 (Kijun-sen): (최고가 + 최저가) / 2 over kijun 기간
        kijun = (
            high.rolling(window=kijun_period, min_periods=kijun_period).max()
            + low.rolling(window=kijun_period, min_periods=kijun_period).min()
        ) / 2.0

        # 선행스팬A (Senkou Span A): (전환선 + 기준선) / 2
        senkou_a = (tenkan + kijun) / 2.0

        # 선행스팬B (Senkou Span B): (최고가 + 최저가) / 2 over senkou 기간
        senkou_b = (
            high.rolling(window=senkou_period, min_periods=senkou_period).max()
            + low.rolling(window=senkou_period, min_periods=senkou_period).min()
        ) / 2.0

        # 클라우드 상단/하단
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)  # 클라우드 상단
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)  # 클라우드 하단

        # 종가가 클라우드 위로 돌파 → Long, 아래로 돌파 → Short
        above_cloud = close > cloud_top  # 종가 > 클라우드 상단
        below_cloud = close < cloud_bottom  # 종가 < 클라우드 하단
        prev_above = above_cloud.shift(1).fillna(False)  # 이전 클라우드 상단 돌파 여부
        prev_below = below_cloud.shift(1).fillna(False)  # 이전 클라우드 하단 돌파 여부

        long_signal = above_cloud & ~prev_above  # 클라우드 상향 돌파
        short_signal = below_cloud & ~prev_below  # 클라우드 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class SuperIchi(BaseIndicator):
    """SuperIchi 지표.

    Ichimoku Cloud의 4가지 조건이 동시에 충족될 때만 시그널을 생성한다.
    조건: 전환선/기준선 크로스, 종가 vs 클라우드, 선행스팬A vs B, 후행스팬 vs 종가.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "SuperIchi"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "tenkan": 9,  # 전환선 기간
            "kijun": 26,  # 기준선 기간
            "senkou": 52,  # 선행스팬B 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """SuperIchi 시그널을 계산한다."""
        p = self._resolve_params(params)
        tenkan_period = int(p["tenkan"])  # 전환선 기간
        kijun_period = int(p["kijun"])  # 기준선 기간
        senkou_period = int(p["senkou"])  # 선행스팬B 기간

        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        # 전환선 (Tenkan-sen)
        tenkan = (
            high.rolling(window=tenkan_period, min_periods=tenkan_period).max()
            + low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
        ) / 2.0

        # 기준선 (Kijun-sen)
        kijun = (
            high.rolling(window=kijun_period, min_periods=kijun_period).max()
            + low.rolling(window=kijun_period, min_periods=kijun_period).min()
        ) / 2.0

        # 선행스팬A, B
        senkou_a = (tenkan + kijun) / 2.0  # 선행스팬A
        senkou_b = (
            high.rolling(window=senkou_period, min_periods=senkou_period).max()
            + low.rolling(window=senkou_period, min_periods=senkou_period).min()
        ) / 2.0  # 선행스팬B

        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)  # 클라우드 상단
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)  # 클라우드 하단

        # 4가지 조건 동시 충족
        # 1) 전환선 > 기준선 (Long) / 전환선 < 기준선 (Short)
        cond1_long = tenkan > kijun  # 전환선이 기준선 위
        cond1_short = tenkan < kijun  # 전환선이 기준선 아래

        # 2) 종가 > 클라우드 상단 (Long) / 종가 < 클라우드 하단 (Short)
        cond2_long = close > cloud_top  # 종가가 클라우드 위
        cond2_short = close < cloud_bottom  # 종가가 클라우드 아래

        # 3) 선행스팬A > 선행스팬B (Long) / 선행스팬A < 선행스팬B (Short)
        cond3_long = senkou_a > senkou_b  # 양운 (상승 클라우드)
        cond3_short = senkou_a < senkou_b  # 음운 (하락 클라우드)

        # 4) 후행스팬(종가의 kijun 기간 전 값) > 종가 kijun 기간 전
        chikou = close  # 후행스팬 = 현재 종가 (kijun 기간 후에 표시)
        past_close = close.shift(kijun_period)  # kijun 기간 전 종가
        cond4_long = chikou > past_close  # 후행스팬 > 과거 종가
        cond4_short = chikou < past_close  # 후행스팬 < 과거 종가

        # 모든 조건 동시 충족
        all_long = cond1_long & cond2_long & cond3_long & cond4_long  # 4조건 모두 Long
        all_short = cond1_short & cond2_short & cond3_short & cond4_short  # 4조건 모두 Short

        prev_all_long = all_long.shift(1).fillna(False)  # 이전 Long 조건
        prev_all_short = all_short.shift(1).fillna(False)  # 이전 Short 조건

        long_signal = all_long & ~prev_all_long  # Long 조건 진입 시점
        short_signal = all_short & ~prev_all_short  # Short 조건 진입 시점

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ChandelierExit(BaseIndicator):
    """Chandelier Exit 지표.

    ATR 기반 추세 추적 스탑으로 방향 전환을 감지한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Chandelier Exit"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "atr_period": 22,  # ATR 기간
            "mult": 3.0,  # ATR 승수
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Chandelier Exit 시그널을 계산한다."""
        p = self._resolve_params(params)
        atr_period = int(p["atr_period"])  # ATR 기간
        mult = float(p["mult"])  # ATR 승수

        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        atr_vals = atr(df, atr_period)  # ATR 값

        # 기간 내 최고가/최저가
        highest = high.rolling(window=atr_period, min_periods=atr_period).max()  # 기간 최고가
        lowest = low.rolling(window=atr_period, min_periods=atr_period).min()  # 기간 최저가

        # Chandelier Exit Long/Short 라인
        long_stop = highest - mult * atr_vals  # Long 스탑 (최고가 - ATR*mult)
        short_stop = lowest + mult * atr_vals  # Short 스탑 (최저가 + ATR*mult)

        n = len(close)  # 데이터 길이
        close_vals = close.values  # 종가 배열
        ls_vals = long_stop.values  # Long 스탑 배열
        ss_vals = short_stop.values  # Short 스탑 배열

        # 트레일링 스탑 계산 (순차 처리)
        final_ls = np.copy(ls_vals)  # 최종 Long 스탑
        final_ss = np.copy(ss_vals)  # 최종 Short 스탑
        direction = np.ones(n, dtype=int)  # 방향 (1=Long, -1=Short)

        for i in range(1, n):
            if np.isnan(ls_vals[i]) or np.isnan(ss_vals[i]):
                final_ls[i] = final_ls[i - 1]
                final_ss[i] = final_ss[i - 1]
                direction[i] = direction[i - 1]
                continue

            # Long 스탑 트레일링
            if close_vals[i - 1] > final_ls[i - 1]:
                final_ls[i] = max(ls_vals[i], final_ls[i - 1])
            else:
                final_ls[i] = ls_vals[i]

            # Short 스탑 트레일링
            if close_vals[i - 1] < final_ss[i - 1]:
                final_ss[i] = min(ss_vals[i], final_ss[i - 1])
            else:
                final_ss[i] = ss_vals[i]

            # 방향 결정
            if close_vals[i] > final_ss[i]:
                direction[i] = 1  # Long
            elif close_vals[i] < final_ls[i]:
                direction[i] = -1  # Short
            else:
                direction[i] = direction[i - 1]

        dir_series = pd.Series(direction, index=df.index)  # 방향 시리즈
        prev_dir = dir_series.shift(1)  # 이전 방향

        long_signal = (dir_series == 1) & (prev_dir == -1)  # Short→Long 전환
        short_signal = (dir_series == -1) & (prev_dir == 1)  # Long→Short 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ParabolicSAR(BaseIndicator):
    """Parabolic SAR 지표.

    표준 포물선 SAR 알고리즘으로 가속 계수를 사용하여 트렌드를 추적한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Parabolic SAR"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "start": 0.02,  # 초기 가속 계수
            "increment": 0.02,  # 가속 계수 증분
            "max": 0.2,  # 최대 가속 계수
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Parabolic SAR 시그널을 계산한다."""
        p = self._resolve_params(params)
        af_start = float(p["start"])  # 초기 가속 계수
        af_inc = float(p["increment"])  # 가속 계수 증분
        af_max = float(p["max"])  # 최대 가속 계수

        high = df["High"].values  # 고가 배열
        low = df["Low"].values  # 저가 배열
        close = df["Close"].values  # 종가 배열
        n = len(close)  # 데이터 길이

        sar = np.full(n, np.nan)  # SAR 값 배열
        direction = np.ones(n, dtype=int)  # 방향 (1=상승, -1=하락)
        af = np.full(n, af_start)  # 가속 계수 배열
        ep = np.full(n, np.nan)  # 극점 (Extreme Point) 배열

        # 초기화: 첫 2개 봉으로 방향 결정
        if n < 2:
            return IndicatorResult(
                long_signal=pd.Series(False, index=df.index),
                short_signal=pd.Series(False, index=df.index),
            )

        # 초기 방향 설정
        if close[1] > close[0]:
            direction[0] = 1  # 상승
            sar[0] = low[0]  # SAR = 저가
            ep[0] = high[0]  # EP = 고가
        else:
            direction[0] = -1  # 하락
            sar[0] = high[0]  # SAR = 고가
            ep[0] = low[0]  # EP = 저가

        for i in range(1, n):
            prev_sar = sar[i - 1]  # 이전 SAR
            prev_af = af[i - 1]  # 이전 가속 계수
            prev_ep = ep[i - 1]  # 이전 극점
            prev_dir = direction[i - 1]  # 이전 방향

            # 새 SAR 계산
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            if prev_dir == 1:  # 상승 트렌드
                # SAR은 이전 2봉의 저가보다 낮아야 함
                new_sar = min(new_sar, low[i - 1])
                if i >= 2:
                    new_sar = min(new_sar, low[i - 2])

                if low[i] < new_sar:  # 반전: 하락으로
                    direction[i] = -1
                    sar[i] = prev_ep  # SAR = 이전 EP
                    ep[i] = low[i]  # EP = 현재 저가
                    af[i] = af_start  # AF 초기화
                else:
                    direction[i] = 1
                    sar[i] = new_sar
                    if high[i] > prev_ep:
                        ep[i] = high[i]  # 새 고점
                        af[i] = min(prev_af + af_inc, af_max)  # AF 증가
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:  # 하락 트렌드
                # SAR은 이전 2봉의 고가보다 높아야 함
                new_sar = max(new_sar, high[i - 1])
                if i >= 2:
                    new_sar = max(new_sar, high[i - 2])

                if high[i] > new_sar:  # 반전: 상승으로
                    direction[i] = 1
                    sar[i] = prev_ep  # SAR = 이전 EP
                    ep[i] = high[i]  # EP = 현재 고가
                    af[i] = af_start  # AF 초기화
                else:
                    direction[i] = -1
                    sar[i] = new_sar
                    if low[i] < prev_ep:
                        ep[i] = low[i]  # 새 저점
                        af[i] = min(prev_af + af_inc, af_max)  # AF 증가
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

        dir_series = pd.Series(direction, index=df.index)  # 방향 시리즈
        prev_dir_s = dir_series.shift(1)  # 이전 방향

        long_signal = (dir_series == 1) & (prev_dir_s == -1)  # 하락→상승 전환
        short_signal = (dir_series == -1) & (prev_dir_s == 1)  # 상승→하락 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class SSLChannel(BaseIndicator):
    """SSL Channel 지표.

    SMA of High와 SMA of Low의 크로스오버로 트렌드를 감지한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "SSL Channel"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 10,  # SMA 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """SSL Channel 시그널을 계산한다."""
        p = self._resolve_params(params)
        period = int(p["period"])  # SMA 기간

        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가

        sma_high = ma(high, period, "SMA")  # 고가 SMA
        sma_low = ma(low, period, "SMA")  # 저가 SMA

        # 종가 기준 방향 결정
        hlv = pd.Series(
            np.where(close > sma_high, 1, np.where(close < sma_low, -1, np.nan)),
            index=df.index,
        )  # 방향 (1=상승, -1=하락)
        hlv = hlv.ffill().fillna(1)  # 전방 채움

        # SSL Up/Down 라인
        ssl_up = pd.Series(
            np.where(hlv > 0, sma_high, sma_low), index=df.index
        )  # SSL 상단 라인
        ssl_down = pd.Series(
            np.where(hlv > 0, sma_low, sma_high), index=df.index
        )  # SSL 하단 라인

        long_signal = crossover(ssl_up, ssl_down)  # SSL Up이 Down 상향 돌파
        short_signal = crossunder(ssl_up, ssl_down)  # SSL Up이 Down 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class HullSuite(BaseIndicator):
    """Hull Suite 지표.

    Hull Moving Average의 방향 변화로 트렌드 전환을 감지한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Hull Suite"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 55,  # HMA 기간
            "mode": "Hma",  # 이동평균 모드 ("Hma", "Thma", "Ehma")
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Hull Suite 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # HMA 기간
        mode = str(p["mode"])  # 이동평균 모드

        close = df["Close"]  # 종가

        if mode.lower() == "hma":
            hull = ma(close, length, "HMA")  # Hull Moving Average
        elif mode.lower() == "thma":
            # Triple HMA: WMA(WMA(close, length/3)*3 - WMA(close, length/2) - WMA(close, length), length)
            l3 = max(int(length / 3), 1)  # length/3
            l2 = max(int(length / 2), 1)  # length/2
            wma3 = ma(close, l3, "WMA")  # WMA(length/3)
            wma2 = ma(close, l2, "WMA")  # WMA(length/2)
            wma_full = ma(close, length, "WMA")  # WMA(length)
            hull = ma(3 * wma3 - wma2 - wma_full, length, "WMA")  # THMA
        elif mode.lower() == "ehma":
            # Exponential HMA: EMA(2*EMA(close, length/2) - EMA(close, length), sqrt(length))
            l2 = max(int(length / 2), 1)  # length/2
            sqrt_l = max(int(np.sqrt(length)), 1)  # sqrt(length)
            ema_half = ma(close, l2, "EMA")  # EMA(length/2)
            ema_full = ma(close, length, "EMA")  # EMA(length)
            hull = ma(2 * ema_half - ema_full, sqrt_l, "EMA")  # EHMA
        else:
            hull = ma(close, length, "HMA")  # 기본값: HMA

        hull_prev = hull.shift(1)  # 이전 Hull 값

        # 방향 변화 감지
        rising = hull > hull_prev  # 상승 중
        falling = hull < hull_prev  # 하락 중
        prev_rising = rising.shift(1).fillna(False)  # 이전 상승 여부
        prev_falling = falling.shift(1).fillna(False)  # 이전 하락 여부

        long_signal = rising & ~prev_rising  # 상승 전환
        short_signal = falling & ~prev_falling  # 하락 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DonchianTrendRibbon(BaseIndicator):
    """Donchian Trend Ribbon 지표.

    Donchian 채널 기반 리본으로 다수 기간의 트렌드 방향을 종합한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Donchian Trend Ribbon"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 15,  # Donchian 채널 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Donchian Trend Ribbon 시그널을 계산한다."""
        p = self._resolve_params(params)
        period = int(p["period"])  # Donchian 채널 기간

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

        # 트렌드 합산이 양수→음수 또는 음수→양수로 전환 시 시그널
        bullish = trend_sum > 0  # 상승 트렌드
        bearish = trend_sum < 0  # 하락 트렌드
        prev_bullish = bullish.shift(1).fillna(False)  # 이전 상승 여부
        prev_bearish = bearish.shift(1).fillna(False)  # 이전 하락 여부

        long_signal = bullish & ~prev_bullish  # 상승 전환
        short_signal = bearish & ~prev_bearish  # 하락 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class TrendlineBreakout(BaseIndicator):
    """Trendline Breakout 지표.

    피봇 고점/저점 기반 추세선을 계산하고 돌파 시 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Trendline Breakout"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # 피봇 룩백 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Trendline Breakout 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # 피봇 룩백 기간

        high = df["High"].values  # 고가 배열
        low = df["Low"].values  # 저가 배열
        close = df["Close"].values  # 종가 배열
        n = len(close)  # 데이터 길이

        long_sig = np.zeros(n, dtype=bool)  # Long 시그널 배열
        short_sig = np.zeros(n, dtype=bool)  # Short 시그널 배열

        # 피봇 고점/저점 감지
        pivot_high_idx = -1  # 마지막 피봇 고점 인덱스
        pivot_high_val = np.nan  # 마지막 피봇 고점 값
        prev_pivot_high_idx = -1  # 이전 피봇 고점 인덱스
        prev_pivot_high_val = np.nan  # 이전 피봇 고점 값

        pivot_low_idx = -1  # 마지막 피봇 저점 인덱스
        pivot_low_val = np.nan  # 마지막 피봇 저점 값
        prev_pivot_low_idx = -1  # 이전 피봇 저점 인덱스
        prev_pivot_low_val = np.nan  # 이전 피봇 저점 값

        for i in range(length, n - length):
            mid = i  # 중심 인덱스

            # 피봇 고점: 중심이 좌우 length 범위 내 최고가
            window_high = high[mid - length : mid + length + 1]  # 윈도우 고가
            if high[mid] == window_high.max() and np.sum(window_high == high[mid]) == 1:
                prev_pivot_high_idx = pivot_high_idx  # 이전 피봇 저장
                prev_pivot_high_val = pivot_high_val
                pivot_high_idx = mid  # 새 피봇 고점
                pivot_high_val = high[mid]

            # 피봇 저점: 중심이 좌우 length 범위 내 최저가
            window_low = low[mid - length : mid + length + 1]  # 윈도우 저가
            if low[mid] == window_low.min() and np.sum(window_low == low[mid]) == 1:
                prev_pivot_low_idx = pivot_low_idx  # 이전 피봇 저장
                prev_pivot_low_val = pivot_low_val
                pivot_low_idx = mid  # 새 피봇 저점
                pivot_low_val = low[mid]

        # 확인 시점 (length 이후)에서 추세선 돌파 체크
        check_idx = mid + length  # 확인 시점
        if check_idx < n:
            # 하락 추세선 돌파 (Long): 두 피봇 고점을 잇는 선 위로 종가 돌파
            if (
                prev_pivot_high_idx >= 0
                and pivot_high_idx > prev_pivot_high_idx
                and not np.isnan(prev_pivot_high_val)
            ):
                slope = (pivot_high_val - prev_pivot_high_val) / (
                    pivot_high_idx - prev_pivot_high_idx
                )  # 추세선 기울기
                trendline_val = pivot_high_val + slope * (
                    check_idx - pivot_high_idx
                )  # 추세선 값
                if (
                    close[check_idx] > trendline_val
                    and close[check_idx - 1] <= trendline_val
                    + slope
                ):
                    long_sig[check_idx] = True  # 상향 돌파

            # 상승 추세선 돌파 (Short): 두 피봇 저점을 잇는 선 아래로 종가 돌파
            if (
                prev_pivot_low_idx >= 0
                and pivot_low_idx > prev_pivot_low_idx
                and not np.isnan(prev_pivot_low_val)
            ):
                slope = (pivot_low_val - prev_pivot_low_val) / (
                    pivot_low_idx - prev_pivot_low_idx
                )  # 추세선 기울기
                trendline_val = pivot_low_val + slope * (
                    check_idx - pivot_low_idx
                )  # 추세선 값
                if (
                    close[check_idx] < trendline_val
                    and close[check_idx - 1] >= trendline_val
                    - slope
                ):
                    short_sig[check_idx] = True  # 하향 돌파

        return IndicatorResult(
            long_signal=pd.Series(long_sig, index=df.index),
            short_signal=pd.Series(short_sig, index=df.index),
        )


# ---------------------------------------------------------------------------
# 2차 배치: 오실레이터/모멘텀 리딩 지표 (13개)
# ---------------------------------------------------------------------------


class TSI(BaseIndicator):
    """True Strength Index (TSI) 지표.

    이중 평활 모멘텀을 사용하여 TSI와 시그널 라인의 크로스오버를 감지한다.
    TSI = EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short) * 100
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "TSI"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "long": 25,  # 장기 평활 기간
            "short": 13,  # 단기 평활 기간
            "signal": 13,  # 시그널 라인 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """TSI 시그널을 계산한다."""
        p = self._resolve_params(params)
        long_period = int(p["long"])  # 장기 평활 기간
        short_period = int(p["short"])  # 단기 평활 기간
        signal_period = int(p["signal"])  # 시그널 라인 기간

        close = df["Close"]  # 종가 시계열
        momentum = close.diff()  # 모멘텀 (종가 변화량)
        abs_momentum = momentum.abs()  # 모멘텀 절대값

        # 이중 평활: EMA(EMA(momentum, long), short)
        double_smoothed = ma(ma(momentum, long_period, "EMA"), short_period, "EMA")
        # 이중 평활 절대값: EMA(EMA(|momentum|, long), short)
        double_smoothed_abs = ma(
            ma(abs_momentum, long_period, "EMA"), short_period, "EMA"
        )

        # TSI = 이중평활모멘텀 / 이중평활절대모멘텀 * 100
        tsi_val = 100.0 * double_smoothed / double_smoothed_abs.replace(0, np.nan)
        # 시그널 라인: EMA(TSI, signal)
        signal_line = ma(tsi_val, signal_period, "EMA")

        long_signal = crossover(tsi_val, signal_line)  # TSI가 시그널 상향 돌파
        short_signal = crossunder(tsi_val, signal_line)  # TSI가 시그널 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class TDFI(BaseIndicator):
    """Trend Direction Force Index (TDFI) 지표.

    MMA 평활 가격의 변화율을 정규화하여 트렌드 방향 힘을 측정한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "TDFI"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "lookback": 13,  # 룩백 기간
            "mma": 13,  # MMA 평활 기간
            "filter_high": 0.05,  # 필터 임계값
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """TDFI 시그널을 계산한다."""
        p = self._resolve_params(params)
        lookback = int(p["lookback"])  # 룩백 기간
        mma_period = int(p["mma"])  # MMA 평활 기간
        filter_high = float(p["filter_high"])  # 필터 임계값

        close = df["Close"]  # 종가 시계열

        # MMA 평활 가격
        mma_val = ma(close, mma_period, "EMA")  # EMA로 MMA 근사
        # 변화량
        mma_diff = mma_val - mma_val.shift(1)  # MMA 변화량
        # 변화량의 절대값 합 (정규화 분모)
        abs_diff_sum = mma_diff.abs().rolling(
            window=lookback, min_periods=1
        ).sum()  # 절대 변화량 합
        # 변화량 합 (정규화 분자)
        diff_sum = mma_diff.rolling(window=lookback, min_periods=1).sum()  # 변화량 합

        # TDFI = 변화량합 / 절대변화량합 (정규화된 힘 지수)
        tdfi = diff_sum / abs_diff_sum.replace(0, np.nan)

        long_signal = tdfi > filter_high  # TDFI가 임계값 초과 → 상승 트렌드
        short_signal = tdfi < -filter_high  # TDFI가 -임계값 미만 → 하락 트렌드

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class Stochastic(BaseIndicator):
    """Stochastic 지표.

    %K와 %D의 크로스오버로 매매 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Stochastic"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # Stochastic 기간
            "smooth_k": 3,  # %K 평활 기간
            "smooth_d": 3,  # %D 평활 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Stochastic 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # Stochastic 기간
        smooth_k = int(p["smooth_k"])  # %K 평활 기간
        smooth_d = int(p["smooth_d"])  # %D 평활 기간

        close = df["Close"]  # 종가
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        # Raw %K 계산
        raw_k = stoch(close, high, low, length)  # Raw Stochastic %K
        # %K = SMA(Raw %K, smooth_k)
        k = ma(raw_k, smooth_k, "SMA")  # 평활된 %K
        # %D = SMA(%K, smooth_d)
        d = ma(k, smooth_d, "SMA")  # %D 시그널 라인

        long_signal = crossover(k, d)  # %K가 %D 상향 돌파
        short_signal = crossunder(k, d)  # %K가 %D 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RSI(BaseIndicator):
    """RSI (Relative Strength Index) 지표.

    RSI와 RSI의 이동평균 크로스오버로 매매 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "RSI"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # RSI 기간
            "ma_length": 14,  # RSI 이동평균 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """RSI 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # RSI 기간
        ma_length = int(p["ma_length"])  # RSI 이동평균 기간

        close = df["Close"]  # 종가

        rsi_val = rsi_func(close, length)  # RSI 계산
        rsi_ma = ma(rsi_val, ma_length, "SMA")  # RSI의 이동평균

        long_signal = crossover(rsi_val, rsi_ma)  # RSI가 MA 상향 돌파
        short_signal = crossunder(rsi_val, rsi_ma)  # RSI가 MA 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ROC(BaseIndicator):
    """Rate of Change (ROC) 지표.

    변화율이 제로라인을 크로스하는 시점에서 매매 시그널을 생성한다.
    ROC = (close - close[length]) / close[length] * 100
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "ROC"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 9,  # ROC 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """ROC 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # ROC 기간

        close = df["Close"]  # 종가
        prev_close = close.shift(length)  # length 기간 전 종가

        # ROC = (현재종가 - 과거종가) / 과거종가 * 100
        roc_val = (close - prev_close) / prev_close.replace(0, np.nan) * 100.0
        zero_line = pd.Series(0.0, index=df.index)  # 제로라인

        long_signal = crossover(roc_val, zero_line)  # ROC가 0 상향 돌파
        short_signal = crossunder(roc_val, zero_line)  # ROC가 0 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class CCI(BaseIndicator):
    """Commodity Channel Index (CCI) 지표.

    CCI가 상/하 밴드를 돌파하는 시점에서 매매 시그널을 생성한다.
    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "CCI"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # CCI 기간
            "upper": 100,  # 상단 밴드
            "lower": -100,  # 하단 밴드
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """CCI 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # CCI 기간
        upper = float(p["upper"])  # 상단 밴드
        lower = float(p["lower"])  # 하단 밴드

        # Typical Price = (High + Low + Close) / 3
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0  # 전형적 가격
        tp_sma = ma(tp, length, "SMA")  # TP의 SMA

        # Mean Deviation = SMA(|TP - SMA(TP)|)
        mean_dev = (tp - tp_sma).abs().rolling(
            window=length, min_periods=length
        ).mean()  # 평균 편차

        # CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
        cci_val = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))

        upper_line = pd.Series(upper, index=df.index)  # 상단 밴드 시리즈
        lower_line = pd.Series(lower, index=df.index)  # 하단 밴드 시리즈

        long_signal = crossover(cci_val, upper_line)  # CCI가 상단 밴드 상향 돌파
        short_signal = crossunder(cci_val, lower_line)  # CCI가 하단 밴드 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class MACD(BaseIndicator):
    """MACD (Moving Average Convergence Divergence) 지표.

    MACD 라인과 시그널 라인의 크로스오버로 매매 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "MACD"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 12,  # 빠른 EMA 기간
            "slow": 26,  # 느린 EMA 기간
            "signal": 9,  # 시그널 라인 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """MACD 시그널을 계산한다."""
        p = self._resolve_params(params)
        fast = int(p["fast"])  # 빠른 EMA 기간
        slow = int(p["slow"])  # 느린 EMA 기간
        signal_period = int(p["signal"])  # 시그널 라인 기간

        close = df["Close"]  # 종가

        # MACD 라인 = EMA(close, fast) - EMA(close, slow)
        macd_line = ma(close, fast, "EMA") - ma(close, slow, "EMA")
        # 시그널 라인 = EMA(MACD, signal)
        signal_line = ma(macd_line, signal_period, "EMA")

        long_signal = crossover(macd_line, signal_line)  # MACD가 시그널 상향 돌파
        short_signal = crossunder(macd_line, signal_line)  # MACD가 시그널 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class BXtrender(BaseIndicator):
    """B-Xtrender 지표.

    RSI+EMA 기반 단기/장기 트렌드 값이 모두 같은 방향일 때 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "B-Xtrender"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "short_l1": 5,  # 단기 RSI 기간
            "short_l2": 20,  # 단기 EMA 평활 기간
            "short_l3": 15,  # 장기 RSI 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """B-Xtrender 시그널을 계산한다."""
        p = self._resolve_params(params)
        short_l1 = int(p["short_l1"])  # 단기 RSI 기간
        short_l2 = int(p["short_l2"])  # 단기 EMA 평활 기간
        short_l3 = int(p["short_l3"])  # 장기 RSI 기간

        close = df["Close"]  # 종가

        # 단기 값: RSI(close, short_l1) - 50, EMA 평활
        short_val = ma(rsi_func(close, short_l1) - 50.0, short_l2, "EMA")  # 단기 트렌드 값
        # 장기 값: RSI(close, short_l3) - 50
        long_val = rsi_func(close, short_l3) - 50.0  # 장기 트렌드 값

        # 단기+장기 모두 양수 → Long, 모두 음수 → Short
        both_positive = (short_val > 0) & (long_val > 0)  # 양수 조건
        both_negative = (short_val < 0) & (long_val < 0)  # 음수 조건

        prev_positive = both_positive.shift(1).fillna(False)  # 이전 양수 조건
        prev_negative = both_negative.shift(1).fillna(False)  # 이전 음수 조건

        long_signal = both_positive & ~prev_positive  # 양수 진입 시점
        short_signal = both_negative & ~prev_negative  # 음수 진입 시점

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class BullBearPowerTrend(BaseIndicator):
    """Bull Bear Power Trend 지표.

    불 파워와 베어 파워를 비교하여 트렌드 방향을 감지한다.
    Bull Power = EMA(High - EMA(Close, period), atr_period)
    Bear Power = EMA(Low - EMA(Close, period), atr_period)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Bull Bear Power Trend"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 50,  # EMA 기간
            "atr": 5,  # 파워 평활 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Bull Bear Power Trend 시그널을 계산한다."""
        p = self._resolve_params(params)
        period = int(p["period"])  # EMA 기간
        atr_period = int(p["atr"])  # 파워 평활 기간

        close = df["Close"]  # 종가
        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        ema_close = ma(close, period, "EMA")  # 종가 EMA

        # Bull Power = EMA(High - EMA(Close), atr_period)
        bull_power = ma(high - ema_close, atr_period, "EMA")  # 불 파워
        # Bear Power = EMA(Low - EMA(Close), atr_period)
        bear_power = ma(low - ema_close, atr_period, "EMA")  # 베어 파워

        # 불/베어 모두 양수 → Long, 모두 음수 → Short
        both_positive = (bull_power > 0) & (bear_power > 0)  # 양수 조건
        both_negative = (bull_power < 0) & (bear_power < 0)  # 음수 조건

        prev_positive = both_positive.shift(1).fillna(False)  # 이전 양수 조건
        prev_negative = both_negative.shift(1).fillna(False)  # 이전 음수 조건

        long_signal = both_positive & ~prev_positive  # 양수 진입 시점
        short_signal = both_negative & ~prev_negative  # 음수 진입 시점

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DPO(BaseIndicator):
    """Detrended Price Oscillator (DPO) 지표.

    디트렌디드 가격 오실레이터가 제로라인을 크로스하는 시점에서 시그널을 생성한다.
    DPO = Close - SMA(Close, period).shift(period // 2 + 1)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "DPO"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 10,  # DPO 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """DPO 시그널을 계산한다."""
        p = self._resolve_params(params)
        period = int(p["period"])  # DPO 기간

        close = df["Close"]  # 종가

        # DPO = Close - SMA(Close, period).shift(period // 2 + 1)
        sma_close = ma(close, period, "SMA")  # 종가 SMA
        shift_amount = period // 2 + 1  # 시프트 양
        dpo_val = close - sma_close.shift(shift_amount)  # DPO 값

        zero_line = pd.Series(0.0, index=df.index)  # 제로라인

        long_signal = crossover(dpo_val, zero_line)  # DPO가 0 상향 돌파
        short_signal = crossunder(dpo_val, zero_line)  # DPO가 0 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class BBOscillator(BaseIndicator):
    """Bollinger Band Oscillator (BB Oscillator) 지표.

    볼린저 밴드 기반 오실레이터가 제로라인을 크로스하는 시점에서 시그널을 생성한다.
    osc = (Close - middle) / (upper - middle) 정규화
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "BB Oscillator"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # 볼린저 밴드 기간
            "stddev": 2.0,  # 표준편차 승수
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """BB Oscillator 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # 볼린저 밴드 기간
        stddev = float(p["stddev"])  # 표준편차 승수

        close = df["Close"]  # 종가

        # 볼린저 밴드 계산
        middle = ma(close, length, "SMA")  # 중간 밴드 (SMA)
        std = close.rolling(window=length, min_periods=length).std()  # 표준편차
        upper = middle + stddev * std  # 상단 밴드
        # 오실레이터: (Close - middle) / (upper - middle)
        band_width = upper - middle  # 밴드 폭
        osc = (close - middle) / band_width.replace(0, np.nan)  # 정규화된 오실레이터

        zero_line = pd.Series(0.0, index=df.index)  # 제로라인

        long_signal = crossover(osc, zero_line)  # 오실레이터가 0 상향 돌파
        short_signal = crossunder(osc, zero_line)  # 오실레이터가 0 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class AwesomeOscillator(BaseIndicator):
    """Awesome Oscillator 지표.

    HL2의 빠른/느린 SMA 차이가 제로라인을 크로스하는 시점에서 시그널을 생성한다.
    AO = SMA(HL2, fast) - SMA(HL2, slow)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Awesome Oscillator"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 5,  # 빠른 SMA 기간
            "slow": 34,  # 느린 SMA 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Awesome Oscillator 시그널을 계산한다."""
        p = self._resolve_params(params)
        fast = int(p["fast"])  # 빠른 SMA 기간
        slow = int(p["slow"])  # 느린 SMA 기간

        # HL2 = (High + Low) / 2
        hl2 = (df["High"] + df["Low"]) / 2.0  # 중간가

        # AO = SMA(HL2, fast) - SMA(HL2, slow)
        ao = ma(hl2, fast, "SMA") - ma(hl2, slow, "SMA")  # Awesome Oscillator 값

        zero_line = pd.Series(0.0, index=df.index)  # 제로라인

        long_signal = crossover(ao, zero_line)  # AO가 0 상향 돌파
        short_signal = crossunder(ao, zero_line)  # AO가 0 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class VolatilityOscillator(BaseIndicator):
    """Volatility Oscillator 지표.

    Close-Open 스파이크가 표준편차 밴드를 돌파하는 시점에서 시그널을 생성한다.
    spike = Close - Open
    upper_band = EMA(spike, length) + stdev(spike, length)
    lower_band = EMA(spike, length) - stdev(spike, length)
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Volatility Oscillator"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 100,  # 오실레이터 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Volatility Oscillator 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # 오실레이터 기간

        # spike = Close - Open
        spike = df["Close"] - df["Open"]  # 스파이크 (종가 - 시가)

        # EMA와 표준편차 계산
        spike_ema = ma(spike, length, "EMA")  # 스파이크 EMA
        spike_std = spike.rolling(
            window=length, min_periods=length
        ).std()  # 스파이크 표준편차

        upper_band = spike_ema + spike_std  # 상단 밴드
        lower_band = spike_ema - spike_std  # 하단 밴드

        long_signal = spike > upper_band  # 스파이크가 상단 밴드 돌파
        short_signal = spike < lower_band  # 스파이크가 하단 밴드 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class TwoEMACross(BaseIndicator):
    """2 EMA Cross 지표.

    빠른 EMA와 느린 EMA의 크로스오버/크로스언더로 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "2 EMA Cross"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 50,  # 빠른 EMA 기간
            "slow": 200,  # 느린 EMA 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """2 EMA Cross 시그널을 계산한다."""
        p = self._resolve_params(params)
        fast = int(p["fast"])  # 빠른 EMA 기간
        slow = int(p["slow"])  # 느린 EMA 기간
        close = df["Close"]  # 종가 시계열

        fast_ema = ma(close, fast, "EMA")  # 빠른 EMA
        slow_ema = ma(close, slow, "EMA")  # 느린 EMA

        long_signal = crossover(fast_ema, slow_ema)  # 빠른 EMA가 느린 EMA 상향 돌파
        short_signal = crossunder(fast_ema, slow_ema)  # 빠른 EMA가 느린 EMA 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ThreeEMACross(BaseIndicator):
    """3 EMA Cross 지표.

    3개 EMA의 정렬 상태 변화(비정렬→정렬 전환)로 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "3 EMA Cross"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "ema1": 9,  # 가장 빠른 EMA 기간
            "ema2": 21,  # 중간 EMA 기간
            "ema3": 55,  # 가장 느린 EMA 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """3 EMA Cross 시그널을 계산한다."""
        p = self._resolve_params(params)
        ema1_len = int(p["ema1"])  # 가장 빠른 EMA 기간
        ema2_len = int(p["ema2"])  # 중간 EMA 기간
        ema3_len = int(p["ema3"])  # 가장 느린 EMA 기간
        close = df["Close"]  # 종가 시계열

        ema1 = ma(close, ema1_len, "EMA")  # 빠른 EMA
        ema2 = ma(close, ema2_len, "EMA")  # 중간 EMA
        ema3 = ma(close, ema3_len, "EMA")  # 느린 EMA

        # 강세 정렬: ema1 > ema2 > ema3
        bullish_aligned = (ema1 > ema2) & (ema2 > ema3)  # 강세 정렬 여부
        # 약세 정렬: ema1 < ema2 < ema3
        bearish_aligned = (ema1 < ema2) & (ema2 < ema3)  # 약세 정렬 여부

        prev_bullish = bullish_aligned.shift(1).fillna(False)  # 이전 강세 정렬
        prev_bearish = bearish_aligned.shift(1).fillna(False)  # 이전 약세 정렬

        # 비정렬→정렬 전환 시점에서 시그널 발생
        long_signal = bullish_aligned & ~prev_bullish  # 강세 정렬 진입
        short_signal = bearish_aligned & ~prev_bearish  # 약세 정렬 진입

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class VWAP(BaseIndicator):
    """VWAP (Volume Weighted Average Price) 지표.

    일별 데이터에서 롤링 VWAP을 계산하여 종가와의 크로스로 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "VWAP"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "anchor": "Session",  # 앵커 기준 (일별 데이터에서는 롤링 방식 사용)
            "length": 20,  # 롤링 VWAP 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """VWAP 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # 롤링 VWAP 기간

        # typical_price = (High + Low + Close) / 3
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0  # 대표가격
        volume = df["Volume"]  # 거래량

        # 롤링 VWAP = SMA(typical_price * volume) / SMA(volume)
        tp_vol = typical_price * volume  # 대표가격 × 거래량
        tp_vol_sum = tp_vol.rolling(window=length, min_periods=length).sum()  # 합계
        vol_sum = volume.rolling(window=length, min_periods=length).sum()  # 거래량 합계
        vwap_line = tp_vol_sum / vol_sum.replace(0, np.nan)  # VWAP 값

        close = df["Close"]  # 종가 시계열

        long_signal = crossover(close, vwap_line)  # 종가가 VWAP 상향 돌파
        short_signal = crossunder(close, vwap_line)  # 종가가 VWAP 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class DMI(BaseIndicator):
    """DMI (Directional Movement Index / ADX) 지표.

    +DI와 -DI의 크로스오버로 방향성 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "DMI"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 14,  # DI 기간
            "adx_smoothing": 14,  # ADX 평활 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """DMI 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # DI 기간

        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        # Directional Movement 계산
        up_move = high.diff()  # 고가 변화량
        down_move = -low.diff()  # 저가 변화량 (부호 반전)

        # +DM: up_move > down_move이고 up_move > 0일 때
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )  # 양의 방향 이동
        # -DM: down_move > up_move이고 down_move > 0일 때
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )  # 음의 방향 이동

        atr_vals = atr(df, length)  # ATR 값

        # +DI = 100 * RMA(+DM) / ATR
        plus_di = 100.0 * ma(plus_dm, length, "RMA") / atr_vals.replace(
            0, np.nan
        )  # +DI
        # -DI = 100 * RMA(-DM) / ATR
        minus_di = 100.0 * ma(minus_dm, length, "RMA") / atr_vals.replace(
            0, np.nan
        )  # -DI

        long_signal = crossover(plus_di, minus_di)  # +DI가 -DI 상향 돌파
        short_signal = crossunder(plus_di, minus_di)  # +DI가 -DI 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class WaddahAttarExplosion(BaseIndicator):
    """Waddah Attar Explosion 지표.

    MACD와 볼린저 밴드 폭을 결합하여 폭발적 움직임 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Waddah Attar Explosion"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "sensitivity": 150,  # 민감도 승수
            "fast": 20,  # 빠른 MACD EMA 기간
            "slow": 40,  # 느린 MACD EMA 기간
            "channel_length": 20,  # 볼린저 밴드 기간
            "mult": 2.0,  # 볼린저 밴드 승수
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Waddah Attar Explosion 시그널을 계산한다."""
        p = self._resolve_params(params)
        sensitivity = float(p["sensitivity"])  # 민감도 승수
        fast = int(p["fast"])  # 빠른 EMA 기간
        slow = int(p["slow"])  # 느린 EMA 기간
        channel_length = int(p["channel_length"])  # BB 기간
        mult = float(p["mult"])  # BB 승수
        close = df["Close"]  # 종가 시계열

        # MACD 계산: (EMA(fast) - EMA(slow)) * sensitivity
        macd_fast = ma(close, fast, "EMA")  # 빠른 EMA
        macd_slow = ma(close, slow, "EMA")  # 느린 EMA
        t1 = (macd_fast - macd_slow) * sensitivity  # 트렌드 값

        # 볼린저 밴드 폭: upper - lower
        bb_mid = close.rolling(window=channel_length, min_periods=channel_length).mean()
        bb_std = close.rolling(window=channel_length, min_periods=channel_length).std()
        e1 = (bb_mid + mult * bb_std) - (bb_mid - mult * bb_std)  # BB 폭 = 2 * mult * std

        # 이전 t1 값으로 진입 전환 감지
        prev_t1 = t1.shift(1)  # 이전 트렌드 값
        prev_long_cond = (prev_t1 > 0) & (prev_t1 > e1.shift(1))  # 이전 롱 조건
        prev_short_cond = (prev_t1 < 0) & (prev_t1.abs() > e1.shift(1))  # 이전 숏 조건

        long_cond = (t1 > 0) & (t1 > e1)  # 현재 롱 조건
        short_cond = (t1 < 0) & (t1.abs() > e1)  # 현재 숏 조건

        long_signal = long_cond & ~prev_long_cond  # 롱 진입 전환
        short_signal = short_cond & ~prev_short_cond  # 숏 진입 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class ChaikinMoneyFlow(BaseIndicator):
    """Chaikin Money Flow 지표.

    자금 흐름 승수와 거래량을 결합하여 매수/매도 압력을 측정한다.
    CMF가 0을 크로스하는 시점에서 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Chaikin Money Flow"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # CMF 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Chaikin Money Flow 시그널을 계산한다."""
        p = self._resolve_params(params)
        length = int(p["length"])  # CMF 기간

        high = df["High"]  # 고가
        low = df["Low"]  # 저가
        close = df["Close"]  # 종가
        volume = df["Volume"]  # 거래량

        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        hl_range = (high - low).replace(0, np.nan)  # 고가-저가 범위 (0 방지)
        mfm = ((close - low) - (high - close)) / hl_range  # 자금 흐름 승수

        # MFV = MFM * Volume
        mfv = mfm * volume  # 자금 흐름 거래량

        # CMF = sum(MFV, length) / sum(Volume, length)
        mfv_sum = mfv.rolling(window=length, min_periods=length).sum()  # MFV 합계
        vol_sum = volume.rolling(window=length, min_periods=length).sum()  # 거래량 합계
        cmf = mfv_sum / vol_sum.replace(0, np.nan)  # CMF 값

        zero_line = pd.Series(0.0, index=df.index)  # 제로라인

        long_signal = crossover(cmf, zero_line)  # CMF가 0 상향 돌파
        short_signal = crossunder(cmf, zero_line)  # CMF가 0 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class VortexIndex(BaseIndicator):
    """Vortex Index 지표.

    VM+와 VM-의 크로스오버로 트렌드 방향 전환 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Vortex Index"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "period": 14,  # Vortex 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Vortex Index 시그널을 계산한다."""
        p = self._resolve_params(params)
        period = int(p["period"])  # Vortex 기간

        high = df["High"]  # 고가
        low = df["Low"]  # 저가

        # VM+ = |High - Low[1]|, VM- = |Low - High[1]|
        vm_plus = (high - low.shift(1)).abs()  # 양의 소용돌이 이동
        vm_minus = (low - high.shift(1)).abs()  # 음의 소용돌이 이동

        # True Range 합계
        tr = true_range(df)  # True Range
        tr_sum = tr.rolling(window=period, min_periods=period).sum()  # TR 합계

        # VI+ = sum(VM+, period) / sum(TR, period)
        vi_plus = vm_plus.rolling(window=period, min_periods=period).sum() / tr_sum.replace(
            0, np.nan
        )  # VI+
        # VI- = sum(VM-, period) / sum(TR, period)
        vi_minus = vm_minus.rolling(window=period, min_periods=period).sum() / tr_sum.replace(
            0, np.nan
        )  # VI-

        long_signal = crossover(vi_plus, vi_minus)  # VI+가 VI- 상향 돌파
        short_signal = crossunder(vi_plus, vi_minus)  # VI+가 VI- 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class STC(BaseIndicator):
    """STC (Schaff Trend Cycle) 지표.

    MACD에 이중 스토캐스틱 평활을 적용하여 0~100 범위의 오실레이터를 생성한다.
    STC가 25를 상향 돌파하면 매수, 75를 하향 돌파하면 매도 시그널.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "STC"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 23,  # 빠른 EMA 기간
            "slow": 50,  # 느린 EMA 기간
            "cycle": 10,  # 스토캐스틱 사이클 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """STC 시그널을 계산한다."""
        p = self._resolve_params(params)
        fast = int(p["fast"])  # 빠른 EMA 기간
        slow = int(p["slow"])  # 느린 EMA 기간
        cycle = int(p["cycle"])  # 스토캐스틱 사이클 기간
        close = df["Close"]  # 종가 시계열
        n = len(close)  # 데이터 길이

        # MACD 값 계산
        macd_val = ma(close, fast, "EMA") - ma(close, slow, "EMA")  # MACD 라인

        macd_arr = macd_val.values  # MACD numpy 배열

        # 1차 스토캐스틱 + EMA 평활 (순차 계산)
        stoch1 = np.full(n, np.nan)  # 1차 스토캐스틱
        pf = np.full(n, np.nan)  # 1차 평활 결과
        factor = 0.5  # 평활 계수

        for i in range(cycle - 1, n):
            window = macd_arr[i - cycle + 1 : i + 1]  # 사이클 윈도우
            valid = window[~np.isnan(window)]  # 유효 값
            if len(valid) == 0:
                continue
            ll = np.min(valid)  # 최저값
            hh = np.max(valid)  # 최고값
            denom = hh - ll  # 분모
            stoch1[i] = ((macd_arr[i] - ll) / denom * 100.0) if denom != 0 else 50.0

            if np.isnan(pf[i - 1]) if i > 0 else True:
                pf[i] = stoch1[i]
            else:
                pf[i] = pf[i - 1] + factor * (stoch1[i] - pf[i - 1])  # EMA 평활

        # 2차 스토캐스틱 + EMA 평활
        stoch2 = np.full(n, np.nan)  # 2차 스토캐스틱
        pff = np.full(n, np.nan)  # 2차 평활 결과 (STC)

        for i in range(cycle - 1, n):
            window = pf[i - cycle + 1 : i + 1]  # 사이클 윈도우
            valid = window[~np.isnan(window)]  # 유효 값
            if len(valid) == 0:
                continue
            ll = np.min(valid)  # 최저값
            hh = np.max(valid)  # 최고값
            denom = hh - ll  # 분모
            stoch2[i] = ((pf[i] - ll) / denom * 100.0) if denom != 0 else 50.0

            if np.isnan(pff[i - 1]) if i > 0 else True:
                pff[i] = stoch2[i]
            else:
                pff[i] = pff[i - 1] + factor * (stoch2[i] - pff[i - 1])  # EMA 평활

        stc_series = pd.Series(pff, index=df.index)  # STC 시계열
        level_25 = pd.Series(25.0, index=df.index)  # 매수 기준선
        level_75 = pd.Series(75.0, index=df.index)  # 매도 기준선

        long_signal = crossover(stc_series, level_25)  # STC가 25 상향 돌파
        short_signal = crossunder(stc_series, level_75)  # STC가 75 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class RangeDetector(BaseIndicator):
    """Range Detector 지표.

    ATR 기반 레인지 감지 및 돌파 시그널을 생성한다.
    평활 ATR 필터로 상/하단 밴드를 구성하고 트렌드 전환을 감지한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Range Detector"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "length": 20,  # 기본 기간
            "mult": 1.0,  # 필터 승수
            "atr_len": 500,  # ATR 평활 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Range Detector 시그널을 계산한다."""
        p = self._resolve_params(params)
        mult = float(p["mult"])  # 필터 승수
        atr_len = int(p["atr_len"])  # ATR 평활 기간
        close = df["Close"]  # 종가 시계열
        n = len(close)  # 데이터 길이

        # 평활 ATR 계산: wper = (atr_len * 2) - 1
        wper = atr_len * 2 - 1  # 평활 기간
        atr_vals = atr(df, 1)  # 1-기간 ATR (= True Range)
        smoothed_atr = ma(atr_vals, wper, "EMA")  # 평활 ATR

        filt = smoothed_atr * mult  # 필터 값

        close_vals = close.values  # 종가 numpy 배열
        filt_vals = filt.values  # 필터 numpy 배열

        # 트렌드 추적 (순차 계산)
        trend = np.zeros(n, dtype=int)  # 트렌드 방향 (1=상승, -1=하락)
        upper = np.full(n, np.nan)  # 상단 밴드
        lower = np.full(n, np.nan)  # 하단 밴드

        for i in range(n):
            if np.isnan(filt_vals[i]):
                continue

            upper[i] = close_vals[i] + filt_vals[i]  # 상단 밴드
            lower[i] = close_vals[i] - filt_vals[i]  # 하단 밴드

            if i == 0:
                trend[i] = 1
                continue

            prev_trend = trend[i - 1]  # 이전 트렌드

            if prev_trend == 1:
                # 상승 트렌드: 하단 밴드는 이전보다 낮아지지 않음
                if not np.isnan(lower[i - 1]):
                    lower[i] = max(lower[i], lower[i - 1])
                if close_vals[i] < lower[i]:
                    trend[i] = -1  # 하락 전환
                else:
                    trend[i] = 1
            else:
                # 하락 트렌드: 상단 밴드는 이전보다 높아지지 않음
                if not np.isnan(upper[i - 1]):
                    upper[i] = min(upper[i], upper[i - 1])
                if close_vals[i] > upper[i]:
                    trend[i] = 1  # 상승 전환
                else:
                    trend[i] = -1

        trend_series = pd.Series(trend, index=df.index)  # 트렌드 시리즈
        prev_trend_series = trend_series.shift(1)  # 이전 트렌드

        long_signal = (trend_series == 1) & (prev_trend_series == -1)  # 하락→상승 전환
        short_signal = (trend_series == -1) & (prev_trend_series == 1)  # 상승→하락 전환

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class HACOLT(BaseIndicator):
    """HACOLT (Heiken-Ashi Candle Oscillator) 지표.

    Heiken-Ashi 종가에 TEMA와 EMA를 적용하여 크로스 시그널을 생성한다.
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "HACOLT"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "tema_period": 55,  # TEMA 기간
            "ema_period": 60,  # EMA 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """HACOLT 시그널을 계산한다."""
        p = self._resolve_params(params)
        tema_period = int(p["tema_period"])  # TEMA 기간
        ema_period = int(p["ema_period"])  # EMA 기간
        n = len(df)  # 데이터 길이

        # Heiken-Ashi 캔들 계산
        ha_close_vals = np.full(n, np.nan)  # HA 종가 배열
        ha_open_vals = np.full(n, np.nan)  # HA 시가 배열

        o = df["Open"].values  # 시가
        h = df["High"].values  # 고가
        l = df["Low"].values  # 저가
        c = df["Close"].values  # 종가

        # ha_close = (O + H + L + C) / 4
        ha_close_vals = (o + h + l + c) / 4.0  # HA 종가

        # ha_open: 순차 계산 (이전 ha_open과 ha_close의 평균)
        ha_open_vals[0] = (o[0] + c[0]) / 2.0  # 첫 번째 HA 시가
        for i in range(1, n):
            ha_open_vals[i] = (ha_open_vals[i - 1] + ha_close_vals[i - 1]) / 2.0

        ha_close = pd.Series(ha_close_vals, index=df.index)  # HA 종가 시리즈

        # TEMA 계산: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
        ema1 = ma(ha_close, tema_period, "EMA")  # 1차 EMA
        ema2 = ma(ema1, tema_period, "EMA")  # 2차 EMA
        ema3 = ma(ema2, tema_period, "EMA")  # 3차 EMA
        tema_val = 3.0 * ema1 - 3.0 * ema2 + ema3  # TEMA 값

        # EMA of HA close
        ema_val = ma(ha_close, ema_period, "EMA")  # HA 종가 EMA

        long_signal = crossover(tema_val, ema_val)  # TEMA가 EMA 상향 돌파
        short_signal = crossunder(tema_val, ema_val)  # TEMA가 EMA 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class WolfpackId(BaseIndicator):
    """Wolfpack Id 지표.

    빠른 EMA와 느린 EMA의 스프레드가 0을 크로스하는 시점에서 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "Wolfpack Id"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "fast": 3,  # 빠른 EMA 기간
            "slow": 8,  # 느린 EMA 기간
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """Wolfpack Id 시그널을 계산한다."""
        p = self._resolve_params(params)
        fast = int(p["fast"])  # 빠른 EMA 기간
        slow = int(p["slow"])  # 느린 EMA 기간
        close = df["Close"]  # 종가 시계열

        fast_ema = ma(close, fast, "EMA")  # 빠른 EMA
        slow_ema = ma(close, slow, "EMA")  # 느린 EMA
        spread = fast_ema - slow_ema  # 스프레드

        zero_line = pd.Series(0.0, index=df.index)  # 제로라인

        long_signal = crossover(spread, zero_line)  # 스프레드가 0 상향 돌파
        short_signal = crossunder(spread, zero_line)  # 스프레드가 0 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )


class QQEMod(BaseIndicator):
    """QQE Mod (Quantitative Qualitative Estimation) 지표.

    RSI를 EMA로 평활한 후 동적 트레일링 밴드를 적용하여 시그널을 생성한다.
    """

    @property
    def name(self) -> str:
        """지표 이름."""
        return "QQE Mod"

    @property
    def default_params(self) -> dict:
        """Pine Script 기본 파라미터."""
        return {
            "rsi_period": 6,  # RSI 기간
            "sf": 5,  # RSI 평활 EMA 기간
            "qqe_factor": 3,  # QQE 팩터 (밴드 폭 승수)
        }

    def calculate(
        self, df: pd.DataFrame, params: dict | None = None
    ) -> IndicatorResult:
        """QQE Mod 시그널을 계산한다."""
        p = self._resolve_params(params)
        rsi_period = int(p["rsi_period"])  # RSI 기간
        sf = int(p["sf"])  # 평활 기간
        qqe_factor = float(p["qqe_factor"])  # QQE 팩터
        close = df["Close"]  # 종가 시계열
        n = len(close)  # 데이터 길이

        # RSI 계산 후 EMA 평활
        rsi_val = rsi_func(close, rsi_period)  # RSI 값
        rsi_smoothed = ma(rsi_val, sf, "EMA")  # 평활 RSI

        rsi_sm = rsi_smoothed.values  # 평활 RSI numpy 배열

        # AtrRsi = |rsi_smoothed - rsi_smoothed[1]|
        atr_rsi = np.abs(np.diff(rsi_sm, prepend=np.nan))  # RSI 변화 절대값
        atr_rsi_series = pd.Series(atr_rsi, index=df.index)

        # 이중 EMA 평활
        ma_atr_rsi = ma(atr_rsi_series, rsi_period * 2 - 1, "EMA")  # 1차 평활
        ma_atr_rsi = ma(ma_atr_rsi, rsi_period * 2 - 1, "EMA")  # 2차 평활

        dar = ma_atr_rsi.values * qqe_factor  # 동적 범위

        # 트레일링 밴드 계산 (순차 처리)
        longband = np.full(n, np.nan)  # 롱 밴드 (하단)
        shortband = np.full(n, np.nan)  # 숏 밴드 (상단)
        trend = np.ones(n, dtype=int)  # 트렌드 방향 (1=상승, -1=하락)

        for i in range(n):
            if np.isnan(rsi_sm[i]) or np.isnan(dar[i]):
                continue

            new_longband = rsi_sm[i] - dar[i]  # 새 롱 밴드
            new_shortband = rsi_sm[i] + dar[i]  # 새 숏 밴드

            if i == 0 or np.isnan(longband[i - 1]):
                longband[i] = new_longband
                shortband[i] = new_shortband
                continue

            # 롱 밴드 조정: 이전보다 높으면 유지 (RSI가 이전 롱밴드 위일 때)
            if rsi_sm[i - 1] > longband[i - 1]:
                longband[i] = max(new_longband, longband[i - 1])
            else:
                longband[i] = new_longband

            # 숏 밴드 조정: 이전보다 낮으면 유지 (RSI가 이전 숏밴드 아래일 때)
            if rsi_sm[i - 1] < shortband[i - 1]:
                shortband[i] = min(new_shortband, shortband[i - 1])
            else:
                shortband[i] = new_shortband

            # 트렌드 결정
            prev_t = trend[i - 1]  # 이전 트렌드
            if prev_t == 1:
                if rsi_sm[i] < longband[i]:
                    trend[i] = -1  # 하락 전환
                else:
                    trend[i] = 1
            else:
                if rsi_sm[i] > shortband[i]:
                    trend[i] = 1  # 상승 전환
                else:
                    trend[i] = -1

        # 시그널: RSI가 트레일링 밴드를 크로스
        rsi_s = pd.Series(rsi_sm, index=df.index)  # 평활 RSI 시리즈
        longband_s = pd.Series(longband, index=df.index)  # 롱 밴드 시리즈
        shortband_s = pd.Series(shortband, index=df.index)  # 숏 밴드 시리즈

        long_signal = crossover(rsi_s, longband_s)  # RSI가 롱밴드 상향 돌파
        short_signal = crossunder(rsi_s, shortband_s)  # RSI가 숏밴드 하향 돌파

        return IndicatorResult(
            long_signal=long_signal.fillna(False),
            short_signal=short_signal.fillna(False),
        )
