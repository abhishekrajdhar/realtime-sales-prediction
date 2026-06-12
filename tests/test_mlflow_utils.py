import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from include.utils.service_discovery import expand_env_placeholders
from include.utils import service_discovery


def test_expand_env_placeholders_uses_env_value(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    config = {
        "mlflow": {
            "tracking_uri": "${MLFLOW_TRACKING_URI:-http://mlflow:5001}",
            "nested": ["${MLFLOW_TRACKING_URI}", "plain"],
        }
    }

    resolved = expand_env_placeholders(config)

    assert resolved["mlflow"]["tracking_uri"] == "http://localhost:5001"
    assert resolved["mlflow"]["nested"] == ["http://localhost:5001", "plain"]


def test_expand_env_placeholders_uses_default_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = {"tracking_uri": "${MLFLOW_TRACKING_URI:-http://localhost:5001}"}

    resolved = expand_env_placeholders(config)

    assert resolved["tracking_uri"] == "http://localhost:5001"


def test_get_mlflow_endpoint_skips_unreachable_env_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
    monkeypatch.setattr(service_discovery, "_probe_http_endpoint", lambda endpoint, path, timeout=2: endpoint == "http://localhost:5001")
    monkeypatch.setattr(service_discovery.os.path, "exists", lambda path: False)
    monkeypatch.delenv("AIRFLOW__CORE__EXECUTOR", raising=False)

    endpoint = service_discovery.get_mlflow_endpoint()

    assert endpoint == "http://localhost:5001"


def test_resolve_mlflow_tracking_uri_falls_back_to_local_store(monkeypatch):
    monkeypatch.setattr(service_discovery, "_probe_http_endpoint", lambda *args, **kwargs: False)
    monkeypatch.delenv("MLFLOW_LOCAL_TRACKING_URI", raising=False)

    uri = service_discovery.resolve_mlflow_tracking_uri("http://mlflow:5001", "http://localhost:5001")

    assert uri == "sqlite:////tmp/mlflow.db"
