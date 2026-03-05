"""지표 공통 유틸리티 함수 모듈.

37개 리딩 지표와 45개+ 확인 지표가 공통으로 사용하는
이동평균, ATR, 크로스오버, RSI 등 핵심 계산 함수를 제공한다.
모든 함수는 pandas/numpy 벡터화 연산을 사용한다.
"""

import numpy as np
import pandas as pd


def ma(
    source: pd.Series,  # 이동평균을 계산할 원본 시계열 데이터
    length: int,  # 이동평균 기간 (양의 정수)
    ma_type: str = "EMA",  # 이동평균 유형 ("SMA", "EMA", "RMA", "WMA", "HMA", "VWMA")
    volume: pd.Series | None = None,  # VWMA 계산 시 필요한 거래량 시계열
) -> pd.Series:  # 계산된 이동평균 시계열
    """다양한 이동평균 계산 (SMA, EMA, RMA, WMA, HMA, VWMA).

    Args:
        source: 원본 시계열 데이터.
        length: 이동평균 기간.
        ma_type: 이동평균 유형.
        volume: VWMA 계산 시 필요한 거래량 데이터.

    Returns:
        계산된 이동평균 pd.Series.

    Raises:
        ValueError: 지원하지 않는 ma_type이 전달된 경우.
        ValueError: VWMA 선택 시 volume이 None인 경우.
    """
    ma_type_upper = ma_type.upper()  # 대소문자 무시

    if ma_type_upper == "SMA":
        return _sma(source, length)
    elif ma_type_upper == "EMA":
        return _ema(source, length)
    elif ma_type_upper == "RMA":
        return _rma(source, length)
    elif ma_type_upper == "WMA":
        return _wma(source, length)
    elif ma_type_upper == "HMA":
        return _hma(source, length)
    elif ma_type_upper == "VWMA":
        if volume is None:
            raise ValueError("VWMA 계산에는 volume 데이터가 필요합니다.")
        return _vwma(source, length, volume)
    else:
        raise ValueError(
            f"지원하지 않는 이동평균 유형: '{ma_type}'. "
            f"지원 유형: SMA, EMA, RMA, WMA, HMA, VWMA"
        )


def _sma(
    source: pd.Series,  # 원본 시계열
    length: int,  # SMA 기간
) -> pd.Series:  # 단순 이동평균 결과
    """단순 이동평균 (Simple Moving Average) 계산."""
    return source.rolling(window=length, min_periods=length).mean()


def _ema(
    source: pd.Series,  # 원본 시계열
    length: int,  # EMA 기간
) -> pd.Series:  # 지수 이동평균 결과
    """지수 이동평균 (Exponential Moving Average) 계산."""
    return source.ewm(span=length, adjust=False).mean()


def _rma(
    source: pd.Series,  # 원본 시계열
    length: int,  # RMA 기간
) -> pd.Series:  # Wilder 평활 이동평균 결과
    """RMA (Running Moving Average) 계산.

    Pine Script의 ta.rma와 동일한 Wilder's smoothing 방식.
    EWM에서 alpha=1/length로 계산한다.
    """
    return source.ewm(alpha=1.0 / length, adjust=False).mean()


def _wma(
    source: pd.Series,  # 원본 시계열
    length: int,  # WMA 기간
) -> pd.Series:  # 가중 이동평균 결과
    """가중 이동평균 (Weighted Moving Average) 계산.

    가중치: [1, 2, 3, ..., length], 최근 값에 더 큰 가중치를 부여한다.
    """
    weights = np.arange(1, length + 1, dtype=float)  # 1부터 length까지의 가중치 배열

    def _weighted_mean(window: np.ndarray) -> float:
        """윈도우 내 가중 평균을 계산한다."""
        return np.dot(window, weights) / weights.sum()

    return source.rolling(window=length, min_periods=length).apply(
        _weighted_mean, raw=True
    )


def _hma(
    source: pd.Series,  # 원본 시계열
    length: int,  # HMA 기간
) -> pd.Series:  # Hull 이동평균 결과
    """Hull 이동평균 (Hull Moving Average) 계산.

    공식: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    """
    half_length = max(int(length / 2), 1)  # n/2 기간 (최소 1)
    sqrt_length = max(int(np.sqrt(length)), 1)  # sqrt(n) 기간 (최소 1)

    wma_half = _wma(source, half_length)  # WMA(n/2)
    wma_full = _wma(source, length)  # WMA(n)

    diff = 2.0 * wma_half - wma_full  # 2 * WMA(n/2) - WMA(n)
    return _wma(diff, sqrt_length)  # WMA(diff, sqrt(n))


def _vwma(
    source: pd.Series,  # 원본 시계열 (가격)
    length: int,  # VWMA 기간
    volume: pd.Series,  # 거래량 시계열
) -> pd.Series:  # 거래량 가중 이동평균 결과
    """거래량 가중 이동평균 (Volume Weighted Moving Average) 계산.

    공식: SMA(source * volume, length) / SMA(volume, length)
    """
    pv = source * volume  # 가격 × 거래량
    pv_sma = pv.rolling(window=length, min_periods=length).mean()  # SMA(가격×거래량)
    v_sma = volume.rolling(window=length, min_periods=length).mean()  # SMA(거래량)
    return pv_sma / v_sma


def true_range(
    df: pd.DataFrame,  # OHLCV DataFrame (High, Low, Close 컬럼 필수)
) -> pd.Series:  # True Range 시계열
    """True Range 계산.

    TR = max(High - Low, |High - Close[t-1]|, |Low - Close[t-1]|)
    첫 번째 행은 High - Low로 계산한다.

    Args:
        df: OHLCV DataFrame.

    Returns:
        True Range pd.Series.
    """
    high = df["High"]  # 고가
    low = df["Low"]  # 저가
    prev_close = df["Close"].shift(1)  # 전일 종가

    tr1 = high - low  # 당일 고가 - 당일 저가
    tr2 = (high - prev_close).abs()  # |당일 고가 - 전일 종가|
    tr3 = (low - prev_close).abs()  # |당일 저가 - 전일 종가|

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # 세 값 중 최대값
    return tr


def atr(
    df: pd.DataFrame,  # OHLCV DataFrame (High, Low, Close 컬럼 필수)
    length: int,  # ATR 기간 (양의 정수)
) -> pd.Series:  # Average True Range 시계열
    """Average True Range 계산.

    Pine Script의 ta.atr과 동일하게 RMA(True Range, length)로 계산한다.

    Args:
        df: OHLCV DataFrame.
        length: ATR 평활 기간.

    Returns:
        ATR pd.Series.
    """
    tr = true_range(df)  # True Range 계산
    return _rma(tr, length)  # RMA로 평활화


def crossover(
    series_a: pd.Series,  # 비교 대상 시계열 A
    series_b: pd.Series,  # 비교 대상 시계열 B
) -> pd.Series:  # 크로스오버 발생 여부 (True/False)
    """크로스오버 감지 (A가 B를 상향 돌파).

    현재 시점에서 A > B이고, 직전 시점에서 A <= B인 경우 True.

    Args:
        series_a: 비교 시계열 A.
        series_b: 비교 시계열 B.

    Returns:
        크로스오버 발생 시점이 True인 boolean pd.Series.
    """
    prev_a = series_a.shift(1)  # A의 직전 값
    prev_b = series_b.shift(1)  # B의 직전 값

    cross = (series_a > series_b) & (prev_a <= prev_b)  # 현재 A>B 이고 이전 A<=B
    return cross.fillna(False)  # NaN은 False로 처리


def crossunder(
    series_a: pd.Series,  # 비교 대상 시계열 A
    series_b: pd.Series,  # 비교 대상 시계열 B
) -> pd.Series:  # 크로스언더 발생 여부 (True/False)
    """크로스언더 감지 (A가 B를 하향 돌파).

    현재 시점에서 A < B이고, 직전 시점에서 A >= B인 경우 True.

    Args:
        series_a: 비교 시계열 A.
        series_b: 비교 시계열 B.

    Returns:
        크로스언더 발생 시점이 True인 boolean pd.Series.
    """
    prev_a = series_a.shift(1)  # A의 직전 값
    prev_b = series_b.shift(1)  # B의 직전 값

    cross = (series_a < series_b) & (prev_a >= prev_b)  # 현재 A<B 이고 이전 A>=B
    return cross.fillna(False)  # NaN은 False로 처리


def stoch(
    close: pd.Series,  # 종가 시계열
    high: pd.Series,  # 고가 시계열
    low: pd.Series,  # 저가 시계열
    length: int,  # Stochastic 기간 (양의 정수)
) -> pd.Series:  # Stochastic %K 값 (0~100)
    """Stochastic %K 계산.

    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)

    Args:
        close: 종가 시계열.
        high: 고가 시계열.
        low: 저가 시계열.
        length: 룩백 기간.

    Returns:
        Stochastic %K pd.Series (0~100 범위).
    """
    lowest_low = low.rolling(window=length, min_periods=length).min()  # 기간 내 최저가
    highest_high = high.rolling(
        window=length, min_periods=length
    ).max()  # 기간 내 최고가

    denom = highest_high - lowest_low  # 분모: 최고가 - 최저가
    # 분모가 0인 경우(가격 변동 없음) NaN 처리하여 0으로 나누기 방지
    stoch_k = 100.0 * (close - lowest_low) / denom.replace(0, np.nan)
    return stoch_k


def rsi(
    source: pd.Series,  # RSI를 계산할 원본 시계열 (보통 종가)
    length: int,  # RSI 기간 (양의 정수)
) -> pd.Series:  # RSI 값 (0~100)
    """RSI (Relative Strength Index) 계산.

    Pine Script 컨벤션에 따라 RMA(Wilder's smoothing)를 사용한다.
    RSI = 100 - 100 / (1 + RS), RS = RMA(gain) / RMA(loss)

    Args:
        source: 원본 시계열 데이터.
        length: RSI 기간.

    Returns:
        RSI pd.Series (0~100 범위).
    """
    delta = source.diff()  # 전일 대비 변화량

    gain = delta.clip(lower=0)  # 상승분 (양수만, 음수는 0)
    loss = (-delta).clip(lower=0)  # 하락분 (절대값, 양수만)

    avg_gain = _rma(gain, length)  # RMA로 평균 상승분 계산
    avg_loss = _rma(loss, length)  # RMA로 평균 하락분 계산

    # avg_loss가 0이면 RS는 무한대 → RSI = 100
    # avg_gain이 0이면 RS = 0 → RSI = 0
    rs = avg_gain / avg_loss.replace(0, np.nan)  # RS = 평균상승 / 평균하락 (0 나누기 방지)
    rsi_values = 100.0 - (100.0 / (1.0 + rs))  # RSI 공식

    # avg_loss == 0 인 경우 RSI = 100 (모든 변동이 상승)
    rsi_values = rsi_values.fillna(
        pd.Series(
            np.where(avg_loss == 0, 100.0, np.nan),
            index=source.index,
        )
    )
    return rsi_values
