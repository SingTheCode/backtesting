"""백테스팅 엔진 모듈.

포지션 시그널 기반 벡터화 수익률 계산, 거래 비용 차감,
성과 지표(총 수익률, CAGR, 최대 낙폭, 샤프/소르티노 비율 등)를 산출한다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """백테스팅 결과를 담는 데이터 클래스."""

    combination_id: str  # 조합 식별자
    total_return: float  # 총 수익률
    cagr: float  # 연환산 수익률 (Compound Annual Growth Rate)
    max_drawdown: float  # 최대 낙폭
    sharpe_ratio: float  # 샤프 비율 (연환산)
    sortino_ratio: float  # 소르티노 비율 (연환산)
    total_trades: int  # 총 거래 횟수 (포지션 변경 횟수)
    win_rate: float  # 승률 (양수 수익 거래 비율)
    strategy_returns: pd.Series  # 일별 전략 수익률 Series
    positions: pd.Series  # 실제 적용된 포지션 Series


class BacktestEngine:
    """벡터화 백테스팅 엔진.

    포지션 시그널과 가격 데이터를 받아 수익률 및 성과 지표를 계산한다.
    시그널은 1일 지연(shift(1))하여 미래 정보 편향을 방지한다.
    """

    def run(
        self,
        positions: pd.Series,
        prices: pd.Series,
        transaction_cost: float = 0.001,
        combination_id: str = "",
    ) -> BacktestResult:
        """포지션 시그널 기반 백테스팅을 수행한다.

        Args:
            positions: 포지션 시그널 Series (1=Long, -1=Short, 0=무포지션).
            prices: 종가(Close) 가격 Series.
            transaction_cost: 포지션 변경 시 차감할 거래 비용 비율 (기본값: 0.1%).
            combination_id: 조합 식별자 문자열.

        Returns:
            BacktestResult: 백테스팅 성과 지표 및 일별 수익률.
        """
        # 시그널 1일 지연: day T 시그널 → day T+1 적용 (미래 정보 편향 방지)
        actual_positions = positions.shift(1).fillna(0).astype(np.int64)

        # 일별 가격 수익률
        price_returns = prices.pct_change().fillna(0.0)

        # 전략 일별 수익률: 포지션 × 가격 수익률
        strategy_returns = actual_positions * price_returns

        # 거래 비용 차감: 포지션 변경 시점에서 비용 차감
        position_changes = actual_positions.diff().fillna(0.0)
        trade_mask = position_changes != 0
        strategy_returns = strategy_returns.copy()
        strategy_returns[trade_mask] -= transaction_cost

        # 성과 지표 계산
        total_return = self._calc_total_return(strategy_returns)
        n_days = len(strategy_returns)
        cagr = self._calc_cagr(total_return, n_days)
        max_drawdown = self._calc_max_drawdown(strategy_returns)
        sharpe_ratio = self._calc_sharpe_ratio(strategy_returns)
        sortino_ratio = self._calc_sortino_ratio(strategy_returns)
        total_trades = self._calc_total_trades(actual_positions)
        win_rate = self._calc_win_rate(strategy_returns, actual_positions)

        return BacktestResult(
            combination_id=combination_id,
            total_return=total_return,
            cagr=cagr,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_trades=total_trades,
            win_rate=win_rate,
            strategy_returns=strategy_returns,
            positions=actual_positions,
        )

    @staticmethod
    def _calc_total_return(strategy_returns: pd.Series) -> float:
        """누적 총 수익률을 계산한다.

        Args:
            strategy_returns: 일별 전략 수익률 Series.

        Returns:
            총 수익률 (예: 0.15 = 15%).
        """
        if len(strategy_returns) == 0:
            return 0.0
        return float((1 + strategy_returns).prod() - 1)

    @staticmethod
    def _calc_cagr(total_return: float, n_days: int) -> float:
        """연환산 수익률(CAGR)을 계산한다.

        Args:
            total_return: 총 수익률.
            n_days: 총 거래일 수.

        Returns:
            연환산 수익률 (252 거래일 기준).
        """
        if n_days <= 0:
            return 0.0
        # 총 수익률이 -100% 이하이면 CAGR 계산 불가
        if total_return <= -1.0:
            return -1.0
        return float((1 + total_return) ** (252 / n_days) - 1)

    @staticmethod
    def _calc_max_drawdown(strategy_returns: pd.Series) -> float:
        """최대 낙폭(Max Drawdown)을 계산한다.

        Args:
            strategy_returns: 일별 전략 수익률 Series.

        Returns:
            최대 낙폭 (음수 값, 예: -0.20 = -20%).
        """
        if len(strategy_returns) == 0:
            return 0.0
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        return float(drawdown.min())

    @staticmethod
    def _calc_sharpe_ratio(strategy_returns: pd.Series) -> float:
        """연환산 샤프 비율을 계산한다 (무위험 수익률 = 0 가정).

        Args:
            strategy_returns: 일별 전략 수익률 Series.

        Returns:
            연환산 샤프 비율. 표준편차가 0이면 0.0 반환.
        """
        if len(strategy_returns) == 0:
            return 0.0
        std = strategy_returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(strategy_returns.mean() / std * np.sqrt(252))

    @staticmethod
    def _calc_sortino_ratio(strategy_returns: pd.Series) -> float:
        """연환산 소르티노 비율을 계산한다 (목표 수익률 = 0 가정).

        하방 편차(downside deviation)만 사용하여 위험 조정 수익률을 측정한다.

        Args:
            strategy_returns: 일별 전략 수익률 Series.

        Returns:
            연환산 소르티노 비율. 하방 편차가 0이면 0.0 반환.
        """
        if len(strategy_returns) == 0:
            return 0.0
        downside = strategy_returns[strategy_returns < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = np.sqrt((downside**2).mean())
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float(strategy_returns.mean() / downside_std * np.sqrt(252))

    @staticmethod
    def _calc_total_trades(actual_positions: pd.Series) -> int:
        """총 거래 횟수를 계산한다 (포지션 변경 횟수).

        초기 0에서 첫 포지션 진입도 거래로 카운트한다.

        Args:
            actual_positions: 실제 적용된 포지션 Series.

        Returns:
            포지션 변경 횟수.
        """
        if len(actual_positions) == 0:
            return 0
        changes = actual_positions.diff().fillna(actual_positions.iloc[0])
        return int((changes != 0).sum())

    @staticmethod
    def _calc_win_rate(
        strategy_returns: pd.Series, actual_positions: pd.Series
    ) -> float:
        """승률을 계산한다 (포지션 보유 중 양수 수익 거래일 비율).

        Args:
            strategy_returns: 일별 전략 수익률 Series.
            actual_positions: 실제 적용된 포지션 Series.

        Returns:
            승률 (0.0 ~ 1.0). 거래가 없으면 0.0 반환.
        """
        # 포지션이 있는 날만 필터링
        active_mask = actual_positions != 0
        active_returns = strategy_returns[active_mask]
        if len(active_returns) == 0:
            return 0.0
        winning = (active_returns > 0).sum()
        return float(winning / len(active_returns))
