"""체크포인트 저장/로딩 모듈.

백테스팅 중간 결과를 pickle로 직렬화하여 파일에 저장하고,
중단 후 재시작 시 이어서 실행할 수 있도록 체크포인트를 복원한다.
"""

import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime

from sp500_backtest.engine.backtest import BacktestResult


@dataclass
class Checkpoint:
    """백테스팅 중간 결과 체크포인트."""

    completed_combinations: list[str] = field(default_factory=list)  # 완료된 조합 ID 목록
    results: list[BacktestResult] = field(default_factory=list)  # 완료된 백테스팅 결과 리스트
    timestamp: datetime = field(default_factory=datetime.now)  # 체크포인트 저장 시각
    total_combinations: int = 0  # 전체 조합 수


def save_checkpoint(checkpoint: Checkpoint, path: str) -> None:
    """체크포인트를 pickle 파일로 저장한다.

    저장 실패 시 경고 메시지를 출력하고 계속 진행한다.

    Args:
        checkpoint: 저장할 체크포인트 객체.
        path: 저장할 파일 경로.
    """
    try:
        # 디렉토리가 없으면 생성
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
    except Exception as e:
        print(f"[WARNING] 체크포인트 저장 실패: {e}")


def load_checkpoint(path: str) -> Checkpoint | None:
    """파일에서 체크포인트를 로딩한다.

    파일이 존재하지 않으면 None을 반환한다.

    Args:
        path: 로딩할 파일 경로.

    Returns:
        Checkpoint 객체 또는 None (파일 없음).
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARNING] 체크포인트 로딩 실패: {e}")
        return None
