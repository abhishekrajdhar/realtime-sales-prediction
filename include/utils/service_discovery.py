"""
Simple service discovery for MLflow and MinIO endpoints
"""

import os
import socket
import logging
import re
from typing import Optional
import urllib.request

logger = logging.getLogger(__name__)


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def expand_env_placeholders(value):
    """Recursively expand ${VAR} and ${VAR:-default} placeholders in config values."""
    if isinstance(value, dict):
        return {key: expand_env_placeholders(val) for key, val in value.items()}
    if isinstance(value, list):
        return [expand_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        env_name = match.group(1)
        default = match.group(2) or ""
        return os.getenv(env_name, default)

    return _ENV_PATTERN.sub(replace, value)


def _probe_http_endpoint(endpoint: str, path: str, timeout: int = 2) -> bool:
    try:
        req = urllib.request.Request(f"{endpoint.rstrip('/')}/{path.lstrip('/')}")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.getcode() == 200
    except Exception:
        return False


def resolve_mlflow_tracking_uri(*candidates: Optional[str]) -> str:
    """
    Resolve the best MLflow tracking URI from explicit candidates.

    Prefers reachable HTTP endpoints, accepts sqlite:// URIs as-is, and falls back
    to a local SQLite tracking store when no remote endpoint is reachable.
    """
    for candidate in candidates:
        if not candidate:
            continue
        if candidate.startswith("sqlite://"):
            return candidate
        if candidate.startswith("http") and _probe_http_endpoint(candidate, "/health"):
            return candidate

    fallback = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "sqlite:////tmp/mlflow.db")
    logger.warning("Falling back to local MLflow tracking URI: %s", fallback)
    return fallback

def get_mlflow_endpoint() -> Optional[str]:
    """Try multiple endpoints to find MLflow"""
    # Check if explicitly set in environment
    env_uri = os.getenv('MLFLOW_TRACKING_URI')
    if env_uri:
        if _probe_http_endpoint(env_uri, "/health"):
            logger.info(f"MLflow is accessible at env URI: {env_uri}")
            return env_uri
        logger.warning(f"MLflow env URI is not reachable: {env_uri}; trying discovery")
    
    # Check if we're in a container by looking for common container indicators
    in_container = os.path.exists('/.dockerenv') or os.environ.get('AIRFLOW__CORE__EXECUTOR')
    
    # Order endpoints based on environment
    if in_container:
        # In container, prioritize service names
        endpoints = [
            'http://mlflow:5001',
            'http://host.docker.internal:5001',
            'http://172.17.0.1:5001',  # Default Docker bridge
            'http://localhost:5001'
        ]
    else:
        # Outside container, prioritize localhost
        endpoints = [
            'http://localhost:5001',
            'http://127.0.0.1:5001',
            'http://host.docker.internal:5001'
        ]

    for endpoint in endpoints:
        if _probe_http_endpoint(endpoint, "/health"):
            logger.info(f"MLflow is accessible at: {endpoint}")
            return endpoint
        logger.debug(f"MLflow not accessible at {endpoint}")

    return resolve_mlflow_tracking_uri()

def get_minio_endpoint() -> Optional[str]:
    """Try multiple endpoints to find MinIO"""
    # Check if explicitly set in environment
    env_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    if env_url:
        if _probe_http_endpoint(env_url, "/minio/health/live"):
            logger.info(f"MinIO is accessible at env URL: {env_url}")
            return env_url
        logger.warning(f"MinIO env URL is not reachable: {env_url}; trying discovery")
    
    # Check if we're in a container
    in_container = os.path.exists('/.dockerenv') or os.environ.get('AIRFLOW__CORE__EXECUTOR')
    
    # Order endpoints based on environment
    if in_container:
        # In container, prioritize service names
        endpoints = [
            'http://minio:9000',
            'http://host.docker.internal:9000',
            'http://172.17.0.1:9000',  # Default Docker bridge
            'http://localhost:9000'
        ]
    else:
        # Outside container, prioritize localhost
        endpoints = [
            'http://localhost:9000',
            'http://127.0.0.1:9000',
            'http://host.docker.internal:9000'
        ]

    for endpoint in endpoints:
        if _probe_http_endpoint(endpoint, "/minio/health/live"):
            logger.info(f"MinIO is accessible at: {endpoint}")
            return endpoint
        logger.debug(f"MinIO not accessible at {endpoint}")

    # If nothing works, return the most likely default based on environment
    default = 'http://minio:9000' if in_container else 'http://localhost:9000'
    logger.warning(f"Could not connect to MinIO, using default: {default}")
    return default


# Backward compatibility
def get_mlflow_uri() -> str:
    """Get MLflow URI (backward compatibility)"""
    return get_mlflow_endpoint()


def get_minio_url() -> str:
    """Get MinIO URL (backward compatibility)"""
    return get_minio_endpoint()
