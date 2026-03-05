"""설정 파일 로더 모듈.

config.yaml을 로딩하여 딕셔너리로 반환한다.
명령줄 인자 없이 기본값으로 실행 가능하도록 구성한다.
"""

from pathlib import Path
from typing import Any

import yaml


# config.yaml 기본 경로 (sp500_backtest 디렉토리 내)
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# config.yaml이 없을 때 사용할 기본 설정값
DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "symbol": "^GSPC",  # 수집 대상 심볼 (예: '^GSPC', 'AAPL')
        "period": "3y",  # 데이터 수집 기간 (예: '1y', '3y', '5y')
    },
    "combination": {
        "max_confirmations": 3,  # 최대 확인 지표 수 (개)
        "max_combinations": 100000,  # 최대 조합 수 한도 (개)
    },
    "backtest": {
        "transaction_cost": 0.001,  # 거래 비용 비율 (0.001 = 0.1%)
        "signal_expiry": 3,  # 시그널 만료 캔들 수 (개)
        "alternate_signal": True,  # 연속 동일 방향 시그널 필터링 여부
    },
    "optimizer": {
        "method": "grid",  # 탐색 방식 ('grid' 또는 'random')
        "random_iterations": 1000,  # 랜덤 서치 반복 횟수 (회)
    },
    "results": {
        "sort_by": "total_return",  # 정렬 기준 ('total_return', 'cagr', 'sharpe_ratio')
        "top_n_display": 20,  # 콘솔 출력 상위 N개 (개)
        "top_n_report": 5,  # QuantStats 리포트 생성 상위 N개 (개)
    },
    "performance": {
        "n_workers": -1,  # 병렬 워커 수 (-1: CPU 코어 수 자동 설정)
        "checkpoint_interval": 100,  # 체크포인트 저장 간격 (조합 수 단위)
    },
    "param_ranges": {
        "ema_cross": {
            "fast_period": {"min": 5, "max": 50, "step": 5},  # 빠른 EMA 기간 탐색 범위
            "slow_period": {"min": 20, "max": 200, "step": 10},  # 느린 EMA 기간 탐색 범위
        },
        "supertrend": {
            "atr_period": {"min": 5, "max": 30, "step": 5},  # ATR 기간 탐색 범위
            "factor": {"min": 1.0, "max": 5.0, "step": 0.5},  # 승수 탐색 범위
        },
        "rsi": {
            "length": {"min": 7, "max": 21, "step": 2},  # RSI 기간 탐색 범위
        },
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """설정 파일을 로딩하여 딕셔너리로 반환한다.

    Args:
        config_path: 설정 파일 경로. None이면 기본 경로(sp500_backtest/config.yaml) 사용.

    Returns:
        설정 딕셔너리. config.yaml이 없으면 DEFAULT_CONFIG 반환.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        return _deep_copy_dict(DEFAULT_CONFIG)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return _deep_copy_dict(DEFAULT_CONFIG)

    return config


def _deep_copy_dict(d: dict) -> dict:
    """중첩 딕셔너리를 깊은 복사한다."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            result[key] = value.copy()
        else:
            result[key] = value
    return result
