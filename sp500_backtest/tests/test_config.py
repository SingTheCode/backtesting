"""설정 파일 로더 단위 테스트.

config.yaml 로딩, 기본값 폴백, 라운드 트립 등을 검증한다.
"""

import tempfile
from pathlib import Path

import yaml
import pytest

from sp500_backtest.config import DEFAULT_CONFIG, load_config


class TestLoadConfig:
    """load_config 함수 단위 테스트."""

    def test_load_default_config_yaml(self):
        """기본 경로의 config.yaml을 정상 로딩한다."""
        config = load_config()
        assert isinstance(config, dict)
        assert "data" in config
        assert "combination" in config
        assert "backtest" in config
        assert "optimizer" in config
        assert "results" in config
        assert "performance" in config
        assert "param_ranges" in config

    def test_default_config_values_match_design(self):
        """로딩된 설정값이 설계 문서의 기본값과 일치한다."""
        config = load_config()
        # 데이터 수집 설정
        assert config["data"]["symbol"] == "^GSPC"
        assert config["data"]["period"] == "3y"
        # 조합 설정
        assert config["combination"]["max_confirmations"] == 3
        assert config["combination"]["max_combinations"] == 100000
        # 백테스팅 설정
        assert config["backtest"]["transaction_cost"] == 0.001
        assert config["backtest"]["signal_expiry"] == 3
        assert config["backtest"]["alternate_signal"] is True
        # 파라미터 최적화 설정
        assert config["optimizer"]["method"] == "grid"
        assert config["optimizer"]["random_iterations"] == 1000
        # 결과 설정
        assert config["results"]["sort_by"] == "total_return"
        assert config["results"]["top_n_display"] == 20
        assert config["results"]["top_n_report"] == 5
        # 성능 설정
        assert config["performance"]["n_workers"] == -1
        assert config["performance"]["checkpoint_interval"] == 100

    def test_param_ranges_structure(self):
        """파라미터 탐색 범위가 min/max/step 구조를 갖는다."""
        config = load_config()
        ranges = config["param_ranges"]
        # ema_cross
        assert ranges["ema_cross"]["fast_period"] == {"min": 5, "max": 50, "step": 5}
        assert ranges["ema_cross"]["slow_period"] == {"min": 20, "max": 200, "step": 10}
        # supertrend
        assert ranges["supertrend"]["atr_period"] == {"min": 5, "max": 30, "step": 5}
        assert ranges["supertrend"]["factor"] == {"min": 1.0, "max": 5.0, "step": 0.5}
        # rsi
        assert ranges["rsi"]["length"] == {"min": 7, "max": 21, "step": 2}

    def test_load_from_custom_path(self):
        """사용자 지정 경로의 YAML 파일을 로딩한다."""
        custom_config = {"data": {"symbol": "AAPL", "period": "1y"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(custom_config, f)
            f.flush()
            loaded = load_config(f.name)
        assert loaded["data"]["symbol"] == "AAPL"
        assert loaded["data"]["period"] == "1y"

    def test_fallback_to_default_when_file_missing(self):
        """존재하지 않는 경로 지정 시 DEFAULT_CONFIG를 반환한다."""
        config = load_config("/nonexistent/path/config.yaml")
        assert config == DEFAULT_CONFIG

    def test_fallback_to_default_when_yaml_empty(self):
        """빈 YAML 파일 로딩 시 DEFAULT_CONFIG를 반환한다."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            config = load_config(f.name)
        assert config == DEFAULT_CONFIG

    def test_default_config_deep_copy(self):
        """DEFAULT_CONFIG 반환 시 원본이 변경되지 않는 깊은 복사를 수행한다."""
        config1 = load_config("/nonexistent/path/config.yaml")
        config1["data"]["symbol"] = "MODIFIED"
        config2 = load_config("/nonexistent/path/config.yaml")
        assert config2["data"]["symbol"] == "^GSPC"

    def test_config_yaml_round_trip(self):
        """설정 딕셔너리를 YAML로 직렬화 후 역직렬화하면 동일한 값을 복원한다."""
        original = DEFAULT_CONFIG
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(original, f, default_flow_style=False)
            f.flush()
            loaded = load_config(f.name)
        assert loaded == original

    def test_no_cli_args_required(self):
        """명령줄 인자 없이 기본값으로 실행 가능하다."""
        # load_config()를 인자 없이 호출하면 정상 동작해야 함
        config = load_config()
        assert isinstance(config, dict)
        assert len(config) > 0
