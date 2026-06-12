import os
import time
import logging
from typing import Dict, Any, Optional
import requests
import yaml

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
from datetime import datetime
import joblib

from .service_discovery import (
    expand_env_placeholders,
    get_mlflow_endpoint,
    get_minio_endpoint,
    resolve_mlflow_tracking_uri,
)

logger = logging.getLogger(__name__)


class MLflowManager:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        # Load config with fallback to package config
        if not os.path.exists(config_path):
            pkg_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "ml_config.yaml"))
            config_path = pkg_config if os.path.exists(pkg_config) else config_path

        try:
            with open(config_path, "r") as f:
                self.config = expand_env_placeholders(yaml.safe_load(f) or {})
        except FileNotFoundError:
            # Best-effort empty config
            logger.warning("MLflow config not found at %s, proceeding with environment/service discovery", config_path)
            self.config = {}

        mlflow_config = self.config.get("mlflow", {})

        # Determine tracking URI: env -> config -> service discovery
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        cfg_uri = mlflow_config.get("tracking_uri")
        svc_uri = None
        try:
            svc_uri = get_mlflow_endpoint()
        except Exception:
            svc_uri = None

        self.tracking_uri = resolve_mlflow_tracking_uri(env_uri, cfg_uri, svc_uri)
        if not self.tracking_uri:
            raise RuntimeError("MLflow tracking URI could not be resolved")

        # Ensure mlflow uses the chosen tracking URI
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info("Set MLflow tracking URI to %s", self.tracking_uri)
        except Exception as e:
            logger.warning("Failed to set MLflow tracking URI to %s: %s", self.tracking_uri, e)

        # Lightweight health check to surface connectivity issues early
        healthy = False
        for attempt in range(6):
            try:
                health_url = self.tracking_uri.rstrip("/") + "/api/2.0/mlflow/experiments/list"
                requests.get(health_url, timeout=3)
                logger.info("MLflow tracking server reachable at %s", self.tracking_uri)
                healthy = True
                break
            except Exception:
                logger.warning("MLflow not reachable at %s (attempt %s/6)", self.tracking_uri, attempt + 1)
                time.sleep(2)
        if not healthy:
            logger.error("Unable to reach MLflow at %s after retries; training may fail", self.tracking_uri)

        # Experiment and registry names (best-effort)
        self.experiment_name = mlflow_config.get("experiment_name", "default")
        self.registry_name = mlflow_config.get("registry_name", "registry")

        # Try to set experiment (non-fatal)
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning("Failed to set experiment %s at %s: %s", self.experiment_name, self.tracking_uri, e)

        # Configure S3/MinIO environment variables
        try:
            minio_ep = get_minio_endpoint()
            if minio_ep:
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_ep
        except Exception:
            logger.debug("Could not obtain MinIO endpoint via service discovery")

        os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
        os.environ.setdefault("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

        # Initialize client with chosen tracking uri
        try:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
        except Exception as e:
            logger.warning("Failed to create MlflowClient for %s: %s", self.tracking_uri, e)
            self.client = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        except Exception:
            pass
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info("Started MLflow run: %s", run.info.run_id)
        return run.info.run_id

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, model_name: str, input_example: Optional[pd.DataFrame] = None,
                  signature: Optional[Any] = None, registered_model_name: Optional[str] = None):
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{model_name}_model.pkl")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path=f"models/{model_name}")
                logger.info("Saved %s model as artifact", model_name)

                metadata = {
                    "model_type": model_name,
                    "framework": type(model).__module__,
                    "class": type(model).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                metadata_path = os.path.join(tmpdir, f"{model_name}_metadata.yaml")
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f)
                mlflow.log_artifact(metadata_path, artifact_path=f"models/{model_name}")
        except Exception as e:
            logger.error("Failed to log model %s: %s", model_name, e)

    def log_artifacts(self, artifact_path: str):
        mlflow.log_artifacts(artifact_path)

    def log_figure(self, figure, artifact_file: str):
        mlflow.log_figure(figure, artifact_file)

    def end_run(self, status: str = "FINISHED"):
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None

        try:
            mlflow.end_run(status=status)
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.warning("mlflow.end_run failed: %s", e)

        # Sync artifacts to S3 after run ends
        if run_id and status == "FINISHED":
            try:
                from .mlflow_s3_utils import MLflowS3Manager
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(run_id)
                logger.info("Synced artifacts to S3 for run %s", run_id)
            except Exception as e:
                logger.warning("Failed to sync artifacts to S3: %s", e)

    def get_best_model(self, metric: str = "rmse", ascending: bool = True) -> Dict[str, Any]:
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                raise ValueError("Experiment not found")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )
            if len(runs) == 0:
                raise ValueError("No runs found in the experiment")
            best_run = runs.iloc[0]
            return {
                "run_id": best_run["run_id"],
                "metrics": {col.replace("metrics.", ""): val
                           for col, val in best_run.items()
                           if col.startswith("metrics.")},
                "params": {col.replace("params.", ""): val
                           for col, val in best_run.items()
                           if col.startswith("params.")}
            }
        except Exception as e:
            logger.error("get_best_model failed: %s", e)
            raise

    def load_model(self, model_uri: str):
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception:
            # Try loading from artifacts as fallback
            try:
                if "runs:/" in model_uri:
                    parts = model_uri.split("/", 2)
                    run_part = parts[1] if len(parts) > 1 else None
                    artifact_path = parts[2] if len(parts) > 2 else ""
                    local_path = mlflow.artifacts.download_artifacts(run_id=run_part, artifact_path=artifact_path)
                    return joblib.load(local_path)
            except Exception as e:
                logger.error("load_model fallback failed for %s: %s", model_uri, e)
            raise ValueError(f"Cannot load model from {model_uri}")

    def register_model(self, run_id: str, model_name: str, artifact_path: str) -> str:
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_version = mlflow.register_model(model_uri, f"{self.registry_name}_{model_name}")
            return model_version.version
        except Exception as e:
            logger.warning("Model registration failed: %s", e)
            return run_id

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        try:
            if self.client:
                self.client.transition_model_version_stage(
                    name=f"{self.registry_name}_{model_name}",
                    version=version,
                    stage=stage
                )
        except Exception:
            logger.warning("Model stage transition not available")

    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not self.client:
                raise ValueError("Mlflow client not available")
            filter_string = f"name='{self.registry_name}_{model_name}'"
            if stage:
                filter_string += f" AND current_stage='{stage}'"
            versions = self.client.search_model_versions(filter_string)
            if not versions:
                raise ValueError(f"No model versions found for {model_name}")
            latest_version = max(versions, key=lambda x: int(x.version))
            return {
                "version": latest_version.version,
                "stage": latest_version.current_stage,
                "run_id": latest_version.run_id,
                "source": latest_version.source
            }
        except Exception:
            # Fallback to finding the best run
            best_model = self.get_best_model()
            return {
                "version": best_model["run_id"],
                "stage": "None",
                "run_id": best_model["run_id"],
                "source": f"runs:/{best_model['run_id']}/models"
            }
