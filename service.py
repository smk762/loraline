from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import boto3
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from metrics import (
    metrics_router,
    observe_job_duration,
    observe_job_enqueued,
    observe_job_status,
    observe_request,
)

LOG = logging.getLogger("self_lora")
logging.basicConfig(level=os.getenv("SELF_LORA_LOG_LEVEL", "INFO"))
TERMINAL_JOB_STATUSES = {"completed", "failed", "dead_letter", "cancelled"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_database_url(value: str) -> str:
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)
    if value.startswith("postgresql://") and "+psycopg" not in value:
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    internal_token: str
    auth_header: str
    auth_scheme: str
    data_dir: Path
    scratch_dir: Path
    public_base_url: str
    allowed_base_models: set[str]
    database_url: str
    redis_url: str
    redis_queue_prefix: str
    lease_ttl_seconds: int
    queue_poll_timeout_seconds: int
    default_lora_duration_minutes: int
    default_chat_wait_seconds: int
    default_lora_cost_gems: int
    default_chat_cost_gems: int
    ollama_base_url: str
    ollama_timeout_seconds: float
    trainer_command: str
    image_backend: str
    image_api_base_url: str
    image_api_key: str
    image_api_generate_path: str
    image_api_img2img_path: str
    image_api_jobs_path: str
    image_api_timeout_seconds: float
    image_api_job_timeout_seconds: float
    image_api_job_poll_interval_seconds: float
    image_api_auth_header: str
    image_api_auth_scheme: str
    video_backend: str
    video_api_base_url: str
    video_api_key: str
    video_api_generate_path: str
    video_api_jobs_path: str
    video_api_timeout_seconds: float
    video_api_job_timeout_seconds: float
    video_api_job_poll_interval_seconds: float
    video_api_auth_header: str
    video_api_auth_scheme: str
    s3_endpoint_url: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket: str
    s3_region: str
    cdn_base_url: str
    chat_worker_count: int
    image_worker_count: int
    lora_worker_count: int
    video_worker_count: int

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("SELF_LORA_DATA_DIR", "/var/lib/self-lora"))
        return cls(
            host=os.getenv("SELF_LORA_HOST", "0.0.0.0"),
            port=int(os.getenv("SELF_LORA_PORT", "8010")),
            internal_token=os.getenv("SELF_LORA_INTERNAL_TOKEN", ""),
            auth_header=os.getenv("SELF_LORA_AUTH_HEADER", "Authorization"),
            auth_scheme=os.getenv("SELF_LORA_AUTH_SCHEME", "Bearer"),
            data_dir=data_dir,
            scratch_dir=Path(os.getenv("SELF_LORA_SCRATCH_DIR", str(data_dir / "scratch"))),
            public_base_url=os.getenv("SELF_LORA_PUBLIC_BASE_URL", "http://127.0.0.1:8010").rstrip("/"),
            allowed_base_models=set(
                _split_csv(os.getenv("SELF_LORA_ALLOWED_BASE_MODELS", "flux_dev,flux_schnell"))
            ),
            database_url=_normalize_database_url(
                os.getenv("SELF_LORA_DATABASE_URL") or os.getenv("DATABASE_URL", "")
            ),
            redis_url=os.getenv("SELF_LORA_REDIS_URL") or os.getenv("REDIS_URL", ""),
            redis_queue_prefix=os.getenv("SELF_LORA_REDIS_QUEUE_PREFIX", "self-lora"),
            lease_ttl_seconds=int(os.getenv("SELF_LORA_LEASE_TTL_SECONDS", "90")),
            queue_poll_timeout_seconds=int(os.getenv("SELF_LORA_QUEUE_POLL_TIMEOUT_SECONDS", "5")),
            default_lora_duration_minutes=int(
                os.getenv("SELF_LORA_DEFAULT_LORA_DURATION_MINUTES", "15")
            ),
            default_chat_wait_seconds=int(os.getenv("SELF_LORA_DEFAULT_CHAT_WAIT_SECONDS", "5")),
            default_lora_cost_gems=int(os.getenv("SELF_LORA_DEFAULT_LORA_COST_GEMS", "0")),
            default_chat_cost_gems=int(os.getenv("SELF_LORA_DEFAULT_CHAT_COST_GEMS", "0")),
            ollama_base_url=os.getenv("SELF_LORA_OLLAMA_BASE_URL", "http://192.168.1.109:11434").rstrip("/"),
            ollama_timeout_seconds=float(os.getenv("SELF_LORA_OLLAMA_TIMEOUT_SECONDS", "120")),
            trainer_command=os.getenv("SELF_LORA_TRAINER_COMMAND", "").strip(),
            image_backend=os.getenv("IMAGE_BACKEND", "").strip(),
            image_api_base_url=os.getenv("IMAGE_API_BASE_URL", "").rstrip("/"),
            image_api_key=os.getenv("IMAGE_API_KEY", ""),
            image_api_generate_path=os.getenv("IMAGE_API_GENERATE_PATH", "/images/generate"),
            image_api_img2img_path=os.getenv("IMAGE_API_IMG2IMG_PATH", "/images/img2img"),
            image_api_jobs_path=os.getenv("IMAGE_API_JOBS_PATH", "/images/jobs"),
            image_api_timeout_seconds=float(os.getenv("IMAGE_API_TIMEOUT_MS", "120000")) / 1000.0,
            image_api_job_timeout_seconds=float(os.getenv("IMAGE_API_JOB_TIMEOUT_MS", "300000")) / 1000.0,
            image_api_job_poll_interval_seconds=float(os.getenv("IMAGE_API_JOB_POLL_INTERVAL_MS", "2000"))
            / 1000.0,
            image_api_auth_header=os.getenv("IMAGE_API_AUTH_HEADER", "Authorization"),
            image_api_auth_scheme=os.getenv("IMAGE_API_AUTH_SCHEME", "Bearer"),
            video_backend=os.getenv("VIDEO_BACKEND", "").strip(),
            video_api_base_url=os.getenv("VIDEO_API_BASE_URL", "").rstrip("/"),
            video_api_key=os.getenv("VIDEO_API_KEY", ""),
            video_api_generate_path=os.getenv("VIDEO_API_GENERATE_PATH", "/videos/generate"),
            video_api_jobs_path=os.getenv("VIDEO_API_JOBS_PATH", "/videos/jobs"),
            video_api_timeout_seconds=float(os.getenv("VIDEO_API_TIMEOUT_MS", "120000")) / 1000.0,
            video_api_job_timeout_seconds=float(os.getenv("VIDEO_API_JOB_TIMEOUT_MS", "600000")) / 1000.0,
            video_api_job_poll_interval_seconds=float(os.getenv("VIDEO_API_JOB_POLL_INTERVAL_MS", "3000"))
            / 1000.0,
            video_api_auth_header=os.getenv("VIDEO_API_AUTH_HEADER", "Authorization"),
            video_api_auth_scheme=os.getenv("VIDEO_API_AUTH_SCHEME", "Bearer"),
            s3_endpoint_url=os.getenv("SELF_LORA_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL", ""),
            s3_access_key_id=os.getenv("SELF_LORA_S3_ACCESS_KEY_ID")
            or os.getenv("S3_ACCESS_KEY_ID", ""),
            s3_secret_access_key=os.getenv("SELF_LORA_S3_SECRET_ACCESS_KEY")
            or os.getenv("S3_SECRET_ACCESS_KEY", ""),
            s3_bucket=os.getenv("SELF_LORA_S3_BUCKET") or os.getenv("S3_BUCKET", ""),
            s3_region=os.getenv("SELF_LORA_S3_REGION") or os.getenv("S3_REGION", "us-east-1"),
            cdn_base_url=os.getenv("SELF_LORA_CDN_BASE_URL") or os.getenv("CDN_BASE_URL", ""),
            chat_worker_count=int(os.getenv("SELF_LORA_CHAT_WORKER_COUNT", "1")),
            image_worker_count=int(os.getenv("SELF_LORA_IMAGE_WORKER_COUNT", "1")),
            lora_worker_count=int(os.getenv("SELF_LORA_LORA_WORKER_COUNT", "1")),
            video_worker_count=int(os.getenv("SELF_LORA_VIDEO_WORKER_COUNT", "1")),
        )


SETTINGS = Settings.from_env()
METADATA = MetaData()

JOBS = Table(
    "jobs",
    METADATA,
    Column("id", String(36), primary_key=True),
    Column("job_kind", String(32), nullable=False),
    Column("job_type", String(32), nullable=False),
    Column("provider_key", String(64), nullable=False),
    Column("provider_job_id", String(255), nullable=True),
    Column("user_id", String(64), nullable=False),
    Column("companion_id", String(64), nullable=True),
    Column("name", String(100), nullable=True),
    Column("base_model", String(100), nullable=True),
    Column("trigger_word", String(100), nullable=True),
    Column("status", String(32), nullable=False),
    Column("progress_pct", Integer, nullable=False, default=0),
    Column("estimated_wait_seconds", Integer, nullable=True),
    Column("estimated_duration_minutes", Integer, nullable=True),
    Column("cost_gems", Integer, nullable=True),
    Column("queue_position", Integer, nullable=True),
    Column("request_json", JSON, nullable=False),
    Column("result_json", JSON, nullable=False, default=dict),
    Column("provider_metadata_json", JSON, nullable=False, default=dict),
    Column("error_message", Text, nullable=True),
    Column("idempotency_key", String(255), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("auth_scope_hash", String(64), nullable=False),
    Column("cancel_requested", Boolean, nullable=False, default=False),
    Column("submitted_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("job_kind", "auth_scope_hash", "idempotency_key", name="uq_exec_jobs_idempotency"),
    schema="exec",
)

JOB_INPUTS = Table(
    "job_inputs",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", String(36), nullable=False),
    Column("role", String(32), nullable=False),
    Column("uri", Text, nullable=False),
    Column("content_type", String(255), nullable=True),
    Column("sha256", String(64), nullable=True),
    Column("size_bytes", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    schema="exec",
)

JOB_ARTIFACTS = Table(
    "job_artifacts",
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("job_id", String(36), nullable=False),
    Column("role", String(32), nullable=False),
    Column("uri", Text, nullable=False),
    Column("content_type", String(255), nullable=True),
    Column("metadata_json", JSON, nullable=False, default=dict),
    Column("created_at", DateTime(timezone=True), nullable=False),
    schema="exec",
)

WORKER_LEASES = Table(
    "worker_leases",
    METADATA,
    Column("worker_id", String(128), primary_key=True),
    Column("queue_name", String(128), nullable=False),
    Column("job_id", String(36), nullable=True),
    Column("heartbeat_at", DateTime(timezone=True), nullable=False),
    Column("lease_expires_at", DateTime(timezone=True), nullable=False),
    schema="exec",
)

PROVIDER_CAPABILITIES = Table(
    "provider_capabilities",
    METADATA,
    Column("provider_key", String(64), primary_key=True),
    Column("capabilities_json", JSON, nullable=False),
    Column("refreshed_at", DateTime(timezone=True), nullable=False),
    schema="exec",
)


class HealthResponse(BaseModel):
    status: str


class CreateLoraJobRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    companion_id: str | None = Field(default=None, max_length=64)
    name: str = Field(..., min_length=1, max_length=100)
    base_model: str = Field(..., min_length=1, max_length=100)
    trigger_word: str = Field(..., min_length=1, max_length=100)
    dataset_urls: list[str] = Field(..., min_length=1, max_length=100)
    training_steps: int = Field(default=1000, ge=100, le=10000)
    callback_url: str | None = Field(default=None, max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1, max_length=32)
    content: str = Field(..., min_length=1)


class CreateChatJobRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    companion_id: str | None = Field(default=None, max_length=64)
    model: str = Field(..., min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=100)
    provider_hint: str | None = Field(default="chat.ollama", max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class CreateImageJobRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    companion_id: str | None = Field(default=None, max_length=64)
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(..., min_length=1, max_length=128)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    num_images: int | None = Field(default=None, ge=1, le=8)
    seed: int | None = None
    nsfw: bool | None = None
    guidance_scale: float | None = Field(default=None, ge=0, le=50)
    num_inference_steps: int | None = Field(default=None, ge=1, le=250)
    lora_id: str | None = Field(default=None, max_length=255)
    lora_url: str | None = Field(default=None, max_length=2000)
    lora_weights_url: str | None = Field(default=None, max_length=2000)
    weights_url: str | None = Field(default=None, max_length=2000)
    lora_strength: float | None = Field(default=None, ge=0, le=2)
    style_preset: str | None = Field(default=None, max_length=128)
    source_image_url: str | None = Field(default=None, max_length=2000)
    strength: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateVideoJobRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    companion_id: str | None = Field(default=None, max_length=64)
    prompt: str = Field(..., min_length=1, max_length=500)
    model: str = Field(..., min_length=1, max_length=128)
    duration: str = Field(..., pattern=r"^\d+s$")
    quality: str = Field(default="standard", pattern=r"^(draft|standard|high)$")
    source_image_url: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AcceptedJobResponse(BaseModel):
    id: str
    status: str
    estimated_wait_seconds: int | None = None
    queue_position: int | None = None
    cost_gems: int | None = None
    estimated_duration_minutes: int | None = None
    status_url: str


class ProviderCapability(BaseModel):
    provider_key: str
    job_kind: str
    job_types: list[str]
    operations: list[str]
    max_concurrency: int
    supports_cancel: bool
    supports_poll: bool
    supports_callbacks: bool = False
    supported_base_models: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResult(BaseModel):
    status: str
    progress_pct: int = 100
    result: dict[str, Any] = Field(default_factory=dict)
    provider_job_id: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class Provider(Protocol):
    capability: ProviderCapability

    def validate(self, payload: dict[str, Any], settings: Settings) -> None:
        ...

    def execute(self, runtime: Runtime, job: dict[str, Any]) -> ProviderResult:
        ...


@dataclass
class AuthContext:
    token_hash: str


@dataclass
class Runtime:
    settings: Settings
    engine: Engine
    redis: Redis
    storage: ArtifactStorage
    providers: dict[str, Provider]
    stop_event: threading.Event
    threads: list[threading.Thread]


class ArtifactStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.settings.s3_endpoint_url:
                raise RuntimeError("S3 endpoint is not configured")
            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key_id,
                aws_secret_access_key=self.settings.s3_secret_access_key,
                region_name=self.settings.s3_region,
            )
        return self._client

    def ensure_bucket(self) -> None:
        if not self.settings.s3_bucket:
            raise RuntimeError("S3 bucket is not configured")
        try:
            self.client.head_bucket(Bucket=self.settings.s3_bucket)
        except Exception:
            kwargs: dict[str, Any] = {"Bucket": self.settings.s3_bucket}
            if self.settings.s3_region and self.settings.s3_region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.settings.s3_region,
                }
            self.client.create_bucket(**kwargs)

    def _key_for(self, job: dict[str, Any], role: str, filename: str) -> str:
        safe_user = job["user_id"].replace("/", "_")
        safe_job_kind = job["job_kind"].replace("/", "_")
        return f"{safe_job_kind}/{safe_user}/{job['id']}/{role}/{filename}"

    def _object_url(self, key: str) -> str:
        if self.settings.cdn_base_url:
            return f"{self.settings.cdn_base_url.rstrip('/')}/{key}"
        return f"s3://{self.settings.s3_bucket}/{key}"

    def upload_file(self, job: dict[str, Any], role: str, local_path: Path) -> dict[str, Any]:
        key = self._key_for(job, role, local_path.name)
        self.client.upload_file(str(local_path), self.settings.s3_bucket, key)
        return {
            "role": role,
            "uri": f"s3://{self.settings.s3_bucket}/{key}",
            "public_url": self._object_url(key),
            "filename": local_path.name,
        }


class LocalLoraTrainerProvider:
    capability = ProviderCapability(
        provider_key="training.lora.local",
        job_kind="lora",
        job_types=["train"],
        operations=["train_lora"],
        max_concurrency=1,
        supports_cancel=True,
        supports_poll=True,
        supported_base_models=sorted(SETTINGS.allowed_base_models),
        metadata={"runner_contract": "subprocess manifest"},
    )

    def validate(self, payload: dict[str, Any], settings: Settings) -> None:
        base_model = payload["base_model"]
        if base_model not in settings.allowed_base_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported base_model '{base_model}'",
            )
        if not payload.get("dataset_urls"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="dataset_urls must contain at least one entry",
            )

    def execute(self, runtime: Runtime, job: dict[str, Any]) -> ProviderResult:
        if not runtime.settings.trainer_command:
            return ProviderResult(
                status="dead_letter",
                progress_pct=0,
                error_message="SELF_LORA_TRAINER_COMMAND is not configured",
            )

        scratch_dir = runtime.settings.scratch_dir / job["id"]
        dataset_dir = scratch_dir / "dataset"
        outputs_dir = scratch_dir / "outputs"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = scratch_dir / "job-spec.json"
        result_path = scratch_dir / "result.json"
        request_payload = dict(job["request_json"])
        downloaded_inputs: list[dict[str, Any]] = []

        try:
            for index, url in enumerate(request_payload["dataset_urls"], start=1):
                local_path = _download_url(url, dataset_dir / f"input-{index}")
                downloaded_inputs.append(
                    {
                        "role": "dataset",
                        "uri": url,
                        "content_type": None,
                        "sha256": _hash_file(local_path),
                        "size_bytes": local_path.stat().st_size,
                    }
                )

            request_metadata = dict(request_payload.get("metadata") or {})
            modality = _resolve_lora_modality(request_payload)
            request_metadata.setdefault("modality", modality)
            request_metadata.setdefault("trainer_provider", _trainer_provider_for_modality(request_metadata))
            spec_payload = {
                "job_id": job["id"],
                "user_id": job["user_id"],
                "companion_id": job["companion_id"],
                "name": job["name"],
                "base_model": job["base_model"],
                "trigger_word": job["trigger_word"],
                "training_steps": request_payload["training_steps"],
                "dataset_dir": str(dataset_dir),
                "dataset_urls": request_payload["dataset_urls"],
                "output_dir": str(outputs_dir),
                "metadata": request_metadata,
            }
            dataset_manifest_hash = _dataset_manifest_hash(request_payload["dataset_urls"])
            config_sha256 = _job_spec_hash(spec_payload)
            manifest_path.write_text(json.dumps(spec_payload, indent=2), encoding="utf-8")
            command = shlex.split(runtime.settings.trainer_command) + [
                "--job-spec",
                str(manifest_path),
                "--result-path",
                str(result_path),
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while process.poll() is None:
                if _is_cancel_requested(runtime.engine, job["id"]):
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return ProviderResult(status="cancelled", progress_pct=0, error_message="Cancelled")
                time.sleep(1)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                error_text = stderr.strip() or stdout.strip() or "Trainer exited with a non-zero status"
                return ProviderResult(status="dead_letter", progress_pct=0, error_message=error_text)
            if not result_path.exists():
                return ProviderResult(
                    status="dead_letter",
                    progress_pct=0,
                    error_message="Trainer completed without writing a result manifest",
                )
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            artifacts = []
            weights_path = _coalesce(
                result_payload.get("weights_path"),
                result_payload.get("adapter_path"),
                result_payload.get("model_path"),
            )
            preview_path = result_payload.get("preview_path")
            weights_sha256: str | None = None
            if weights_path:
                weights_sha256 = _hash_file_prefixed(Path(weights_path))
            if weights_path:
                artifacts.append(runtime.storage.upload_file(job, "weights", Path(weights_path)))
            if preview_path:
                artifacts.append(runtime.storage.upload_file(job, "preview", Path(preview_path)))
            provider_metadata = dict(result_payload.get("metadata") or {})
            provider_metadata["downloaded_inputs"] = downloaded_inputs
            provider_metadata["dataset_manifest_hash"] = dataset_manifest_hash
            provider_metadata["weights_sha256"] = weights_sha256
            provider_metadata["config_sha256"] = config_sha256
            trainer_config_path = result_payload.get("config_path")
            if trainer_config_path:
                config_path = Path(trainer_config_path)
                if config_path.exists():
                    provider_metadata["config_sha256"] = _hash_file_prefixed(config_path)
                else:
                    LOG.warning("Trainer returned config_path that does not exist: %s", trainer_config_path)
            if provider_metadata["weights_sha256"] is None:
                LOG.warning("LoRA job %s completed without weights_sha256", job["id"])
            result = dict(result_payload.get("result") or {})
            if artifacts:
                by_role = {artifact["role"]: artifact for artifact in artifacts}
                if "weights" in by_role:
                    result.setdefault("weights_url", by_role["weights"]["public_url"])
                    result.setdefault("adapter_url", by_role["weights"]["public_url"])
                    result.setdefault("model_url", by_role["weights"]["public_url"])
                if "preview" in by_role:
                    result.setdefault("preview_url", by_role["preview"]["public_url"])
            return ProviderResult(
                status=result_payload.get("status", "completed"),
                progress_pct=int(result_payload.get("progress_pct", 100)),
                result=result,
                provider_job_id=result_payload.get("provider_job_id"),
                provider_metadata=provider_metadata,
                error_message=result_payload.get("error_message"),
                artifacts=artifacts,
            )
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)


class OllamaChatProvider:
    capability = ProviderCapability(
        provider_key="chat.ollama",
        job_kind="chat",
        job_types=["completion"],
        operations=["chat_completion"],
        max_concurrency=4,
        supports_cancel=False,
        supports_poll=True,
        metadata={"base_url_env": "SELF_LORA_OLLAMA_BASE_URL"},
    )

    def validate(self, payload: dict[str, Any], settings: Settings) -> None:
        if not payload.get("messages"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="messages must contain at least one entry",
            )

    def execute(self, runtime: Runtime, job: dict[str, Any]) -> ProviderResult:
        payload = dict(job["request_json"])
        request_body = {
            "model": payload["model"],
            "messages": payload["messages"],
            "stream": False,
            "options": payload.get("options") or {},
        }
        with httpx.Client(timeout=runtime.settings.ollama_timeout_seconds) as client:
            response = client.post(f"{runtime.settings.ollama_base_url}/api/chat", json=request_body)
            response.raise_for_status()
            body = response.json()
        result = {
            "model": body.get("model", payload["model"]),
            "message": body.get("message"),
            "content": (body.get("message") or {}).get("content"),
            "done_reason": body.get("done_reason"),
            "total_duration": body.get("total_duration"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
        }
        return ProviderResult(status="completed", progress_pct=100, result=result)


def _resolve_lora_modality(payload: dict[str, Any]) -> str:
    metadata = dict(payload.get("metadata") or {})
    modality = str(metadata.get("modality") or "").strip().lower()
    return "video" if modality == "video" else "image"


def _trainer_provider_for_modality(metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("trainer_provider") or "").strip()
    if explicit:
        return explicit
    default_provider = os.getenv("SELF_LORA_TRAINER_PROVIDER", "mock-image-lora").strip() or "mock-image-lora"
    modality = str(metadata.get("modality") or "image").strip().lower()
    if default_provider == "external-command":
        return "external-video-command" if modality == "video" else "external-image-command"
    if modality == "video" and default_provider in {"mock-image-lora", "external-image-command"}:
        return "external-video-command"
    return default_provider


def _join_url(base_url: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _build_api_headers(auth_header: str, auth_scheme: str, api_key: str, *, idempotency_key: str) -> dict[str, str]:
    headers = {"Idempotency-Key": idempotency_key}
    if api_key:
        headers[auth_header] = f"{auth_scheme} {api_key}"
    return headers


def _poll_remote_job(
    client: httpx.Client,
    *,
    status_url: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started_at = time.monotonic()
    terminal_statuses = {"completed", "failed", "dead_letter", "cancelled"}
    while True:
        response = client.get(status_url)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") in terminal_statuses:
            return payload
        if time.monotonic() - started_at > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for remote job at {status_url}")
        time.sleep(poll_interval_seconds)


def _accepted_job_id(payload: dict[str, Any]) -> str:
    job_id = payload.get("id")
    if not job_id:
        raise RuntimeError("Remote service accepted the job without returning an id")
    return str(job_id)


class SelfHostedImageProvider:
    capability = ProviderCapability(
        provider_key="image.selfhosted",
        job_kind="image",
        job_types=["generate"],
        operations=["generate_image", "image_to_image"],
        max_concurrency=1,
        supports_cancel=False,
        supports_poll=True,
        metadata={"base_url_env": "IMAGE_API_BASE_URL"},
    )

    def validate(self, payload: dict[str, Any], settings: Settings) -> None:
        if not settings.image_api_base_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="IMAGE_API_BASE_URL is not configured",
            )
        if settings.image_backend and settings.image_backend != "selfhosted":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported IMAGE_BACKEND '{settings.image_backend}'",
            )
        if payload.get("source_image_url") and not settings.image_api_img2img_path:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="IMAGE_API_IMG2IMG_PATH is not configured",
            )

    def execute(self, runtime: Runtime, job: dict[str, Any]) -> ProviderResult:
        payload = dict(job["request_json"])
        request_body = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "user_id",
                "companion_id",
                "metadata",
            }
            and value is not None
        }
        submit_path = (
            runtime.settings.image_api_img2img_path
            if payload.get("source_image_url")
            else runtime.settings.image_api_generate_path
        )
        submit_url = _join_url(runtime.settings.image_api_base_url, submit_path)
        headers = _build_api_headers(
            runtime.settings.image_api_auth_header,
            runtime.settings.image_api_auth_scheme,
            runtime.settings.image_api_key,
            idempotency_key=job["id"],
        )
        with httpx.Client(timeout=runtime.settings.image_api_timeout_seconds) as client:
            response = client.post(submit_url, json=request_body, headers=headers)
            response.raise_for_status()
            accepted = response.json()
            provider_job_id = _accepted_job_id(accepted)
            final_payload = _poll_remote_job(
                client,
                status_url=_join_url(runtime.settings.image_api_base_url, f"{runtime.settings.image_api_jobs_path}/{provider_job_id}"),
                poll_interval_seconds=runtime.settings.image_api_job_poll_interval_seconds,
                timeout_seconds=runtime.settings.image_api_job_timeout_seconds,
            )
        result = dict(final_payload.get("result") or {})
        images = list(result.get("images") or [])
        if images:
            result.setdefault("primary_image_url", images[0].get("url"))
            result.setdefault("thumbnail_url", images[0].get("thumbnail_url"))
        return ProviderResult(
            status=final_payload.get("status", "completed"),
            progress_pct=int(final_payload.get("progress_pct", 100)),
            result=result,
            provider_job_id=provider_job_id,
            provider_metadata={
                "provider": self.capability.provider_key,
                "upstream_job_id": provider_job_id,
                "base_url": runtime.settings.image_api_base_url,
                "submit_path": submit_path,
            },
            error_message=final_payload.get("error_message"),
        )


class SelfHostedVideoProvider:
    capability = ProviderCapability(
        provider_key="video.selfhosted",
        job_kind="video",
        job_types=["generate"],
        operations=["generate_video", "image_to_video"],
        max_concurrency=1,
        supports_cancel=False,
        supports_poll=True,
        metadata={"base_url_env": "VIDEO_API_BASE_URL"},
    )

    def validate(self, payload: dict[str, Any], settings: Settings) -> None:
        if not settings.video_api_base_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="VIDEO_API_BASE_URL is not configured",
            )
        if settings.video_backend and settings.video_backend != "selfhosted":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported VIDEO_BACKEND '{settings.video_backend}'",
            )

    def execute(self, runtime: Runtime, job: dict[str, Any]) -> ProviderResult:
        payload = dict(job["request_json"])
        request_body = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "user_id",
                "companion_id",
                "metadata",
            }
            and value is not None
        }
        submit_url = _join_url(runtime.settings.video_api_base_url, runtime.settings.video_api_generate_path)
        headers = _build_api_headers(
            runtime.settings.video_api_auth_header,
            runtime.settings.video_api_auth_scheme,
            runtime.settings.video_api_key,
            idempotency_key=job["id"],
        )
        with httpx.Client(timeout=runtime.settings.video_api_timeout_seconds) as client:
            response = client.post(submit_url, json=request_body, headers=headers)
            response.raise_for_status()
            accepted = response.json()
            provider_job_id = _accepted_job_id(accepted)
            final_payload = _poll_remote_job(
                client,
                status_url=_join_url(runtime.settings.video_api_base_url, f"{runtime.settings.video_api_jobs_path}/{provider_job_id}"),
                poll_interval_seconds=runtime.settings.video_api_job_poll_interval_seconds,
                timeout_seconds=runtime.settings.video_api_job_timeout_seconds,
            )
        result = dict(final_payload.get("result") or {})
        return ProviderResult(
            status=final_payload.get("status", "completed"),
            progress_pct=int(final_payload.get("progress_pct", 100)),
            result=result,
            provider_job_id=provider_job_id,
            provider_metadata={
                "provider": self.capability.provider_key,
                "upstream_job_id": provider_job_id,
                "base_url": runtime.settings.video_api_base_url,
                "submit_path": runtime.settings.video_api_generate_path,
            },
            error_message=final_payload.get("error_message"),
        )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_prefixed_from_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_prefixed_from_text(value: str) -> str:
    return _sha256_prefixed_from_bytes(value.encode("utf-8"))


def _hash_file_prefixed(path: Path) -> str:
    return f"sha256:{_hash_file(path)}"


def _dataset_manifest_hash(dataset_urls: list[str]) -> str:
    canonical = json.dumps(sorted(dataset_urls), separators=(",", ":"), ensure_ascii=True)
    return _sha256_prefixed_from_text(canonical)


def _job_spec_hash(spec_payload: dict[str, Any]) -> str:
    canonical = json.dumps(spec_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_prefixed_from_text(canonical)


def _deterministic_trainer_backend(job: dict[str, Any]) -> str:
    modality = _resolve_lora_modality(dict(job.get("request_json") or {}))
    if modality == "image":
        return "onetrainer"
    raise RuntimeError(f"No deterministic trainer_backend mapping for modality '{modality}'")


def _resolve_trainer_backend(job: dict[str, Any], provider_metadata: dict[str, Any]) -> str:
    explicit = str(provider_metadata.get("trainer_backend") or "").strip()
    if explicit:
        return explicit
    return _deterministic_trainer_backend(job)


def _download_url(url: str, destination_prefix: Path) -> Path:
    with httpx.Client(timeout=300) as client, client.stream("GET", url) as response:
        response.raise_for_status()
        guessed_name = url.rstrip("/").split("/")[-1] or destination_prefix.name
        destination = destination_prefix.with_name(guessed_name)
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)
    return destination


def _queue_name(settings: Settings, job_kind: str) -> str:
    return f"{settings.redis_queue_prefix}:queue:{job_kind}"


def _job_tier(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        tier = metadata.get("tier")
        if tier is not None:
            tier_value = str(tier).strip()
            if tier_value:
                return tier_value
    return "standard"


def _build_request_hash(job_kind: str, auth_scope_hash: str, payload: dict[str, Any]) -> str:
    return _hash_text(_json_dumps({"job_kind": job_kind, "auth_scope_hash": auth_scope_hash, "payload": payload}))


def _estimated_wait_seconds(settings: Settings, job_kind: str, queue_position: int) -> int:
    if job_kind == "chat":
        return max(settings.default_chat_wait_seconds, queue_position * settings.default_chat_wait_seconds)
    if job_kind == "lora":
        return max(1, queue_position * settings.default_lora_duration_minutes * 60)
    return 0


def _estimated_duration_minutes(settings: Settings, payload: dict[str, Any]) -> int:
    training_steps = int(payload.get("training_steps", 1000))
    factor = max(1.0, training_steps / 1000.0)
    return max(1, round(settings.default_lora_duration_minutes * factor))


def _cost_gems(settings: Settings, job_kind: str) -> int:
    if job_kind == "chat":
        return settings.default_chat_cost_gems
    if job_kind == "lora":
        return settings.default_lora_cost_gems
    return 0


def _provider_from_job(runtime: Runtime, job: dict[str, Any]) -> Provider:
    provider_key = job["provider_key"]
    provider = runtime.providers.get(provider_key)
    if provider is None:
        raise RuntimeError(f"Unknown provider '{provider_key}'")
    return provider


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _normalize_lora_result(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job.get("result_json") or {})
    provider_metadata = dict(job.get("provider_metadata_json") or {})
    weights_url = _coalesce(result.get("weights_url"), result.get("adapter_url"), result.get("model_url"))
    preview_url = result.get("preview_url")
    response = {
        "id": job["id"],
        "status": job["status"],
        "progress_pct": job["progress_pct"],
        "preview_url": preview_url,
        "weights_url": weights_url,
        "adapter_url": result.get("adapter_url"),
        "model_url": result.get("model_url"),
        "error_message": job["error_message"],
        "created_at": _isoformat(job["created_at"]),
        "updated_at": _isoformat(job["updated_at"]),
        "estimated_duration_minutes": job["estimated_duration_minutes"],
        "cost_gems": job["cost_gems"],
        "provider_job_id": job["provider_job_id"],
        "trainer_backend": provider_metadata.get("trainer_backend"),
        "base_model": job.get("base_model"),
        "dataset_manifest_hash": provider_metadata.get("dataset_manifest_hash"),
        "weights_sha256": provider_metadata.get("weights_sha256"),
        "config_sha256": provider_metadata.get("config_sha256"),
        "result": result,
    }
    return response


def _normalize_chat_result(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job.get("result_json") or {})
    return {
        "id": job["id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "progress_pct": job["progress_pct"],
        "content": result.get("content"),
        "message": result.get("message"),
        "model": result.get("model") or job["request_json"].get("model"),
        "error_message": job["error_message"],
        "created_at": _isoformat(job["created_at"]),
        "updated_at": _isoformat(job["updated_at"]),
    }


def _normalize_image_result(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job.get("result_json") or {})
    images = list(result.get("images") or [])
    first_image = images[0] if images else {}
    return {
        "id": job["id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "progress_pct": job["progress_pct"],
        "images": images,
        "primary_image_url": result.get("primary_image_url") or first_image.get("url"),
        "thumbnail_url": result.get("thumbnail_url") or first_image.get("thumbnail_url"),
        "model_used": result.get("model_used") or job["request_json"].get("model"),
        "seed": result.get("seed"),
        "duration_ms": result.get("duration_ms"),
        "cost_gems": result.get("cost_gems", job["cost_gems"]),
        "provider_job_id": job["provider_job_id"],
        "error_message": job["error_message"],
        "created_at": _isoformat(job["created_at"]),
        "updated_at": _isoformat(job["updated_at"]),
        "result": result,
    }


def _normalize_video_result(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job.get("result_json") or {})
    return {
        "id": job["id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "progress_pct": job["progress_pct"],
        "video_url": result.get("video_url"),
        "thumbnail_url": result.get("thumbnail_url"),
        "model_used": result.get("model_used") or job["request_json"].get("model"),
        "duration_seconds": result.get("duration_seconds"),
        "duration_ms": result.get("duration_ms"),
        "cost_gems": result.get("cost_gems", job["cost_gems"]),
        "provider_job_id": job["provider_job_id"],
        "error_message": job["error_message"],
        "created_at": _isoformat(job["created_at"]),
        "updated_at": _isoformat(job["updated_at"]),
        "result": result,
    }


def _accepted_response(runtime: Runtime, job: dict[str, Any]) -> AcceptedJobResponse:
    return AcceptedJobResponse(
        id=job["id"],
        status=job["status"],
        estimated_wait_seconds=job["estimated_wait_seconds"],
        queue_position=job["queue_position"],
        cost_gems=job["cost_gems"],
        estimated_duration_minutes=job["estimated_duration_minutes"],
        status_url=f"{runtime.settings.public_base_url}/v1/{job['job_kind']}/jobs/{job['id']}",
    )


def _validate_config(settings: Settings) -> None:
    missing = []
    if not settings.internal_token:
        missing.append("SELF_LORA_INTERNAL_TOKEN")
    if not settings.database_url:
        missing.append("SELF_LORA_DATABASE_URL or DATABASE_URL")
    if not settings.redis_url:
        missing.append("SELF_LORA_REDIS_URL or REDIS_URL")
    if not settings.s3_bucket:
        missing.append("SELF_LORA_S3_BUCKET or S3_BUCKET")
    if not settings.s3_endpoint_url:
        missing.append("SELF_LORA_S3_ENDPOINT_URL or S3_ENDPOINT_URL")
    if missing:
        raise RuntimeError(f"Missing required configuration: {', '.join(missing)}")


@contextmanager
def _connection(runtime: Runtime) -> Any:
    with runtime.engine.begin() as conn:
        yield conn


def _fetch_job(runtime: Runtime, job_id: str) -> dict[str, Any]:
    with _connection(runtime) as conn:
        row = conn.execute(select(JOBS).where(JOBS.c.id == job_id)).mappings().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _row_to_dict(row)


def _store_job_inputs(conn: Connection, job_id: str, payload: dict[str, Any]) -> None:
    now = _utcnow()
    rows = []
    if "dataset_urls" in payload:
        rows.extend(
            {
                "job_id": job_id,
                "role": "dataset",
                "uri": url,
                "content_type": None,
                "sha256": None,
                "size_bytes": None,
                "created_at": now,
            }
            for url in payload["dataset_urls"]
        )
    source_image_url = payload.get("source_image_url")
    if source_image_url:
        rows.append(
            {
                "job_id": job_id,
                "role": "source_image",
                "uri": source_image_url,
                "content_type": None,
                "sha256": None,
                "size_bytes": None,
                "created_at": now,
            }
        )
    if rows:
        conn.execute(insert(JOB_INPUTS), rows)


def _insert_job(
    runtime: Runtime,
    *,
    job_kind: str,
    job_type: str,
    provider_key: str,
    auth_context: AuthContext,
    idempotency_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request_hash = _build_request_hash(job_kind, auth_context.token_hash, payload)
    provider = runtime.providers[provider_key]
    provider.validate(payload, runtime.settings)
    now = _utcnow()
    queue_position = int(runtime.redis.llen(_queue_name(runtime.settings, job_kind))) + 1
    row = {
        "id": str(uuid.uuid4()),
        "job_kind": job_kind,
        "job_type": job_type,
        "provider_key": provider_key,
        "provider_job_id": None,
        "user_id": payload["user_id"],
        "companion_id": payload.get("companion_id"),
        "name": payload.get("name"),
        "base_model": payload.get("base_model"),
        "trigger_word": payload.get("trigger_word"),
        "status": "queued",
        "progress_pct": 0,
        "estimated_wait_seconds": _estimated_wait_seconds(runtime.settings, job_kind, queue_position),
        "estimated_duration_minutes": _estimated_duration_minutes(runtime.settings, payload)
        if job_kind == "lora"
        else None,
        "cost_gems": _cost_gems(runtime.settings, job_kind),
        "queue_position": queue_position,
        "request_json": payload,
        "result_json": {},
        "provider_metadata_json": {},
        "error_message": None,
        "idempotency_key": idempotency_key,
        "request_hash": request_hash,
        "auth_scope_hash": auth_context.token_hash,
        "cancel_requested": False,
        "submitted_at": now,
        "started_at": None,
        "finished_at": None,
        "created_at": now,
        "updated_at": now,
    }
    try:
        with _connection(runtime) as conn:
            conn.execute(insert(JOBS).values(**row))
            _store_job_inputs(conn, row["id"], payload)
    except IntegrityError:
        with _connection(runtime) as conn:
            existing = conn.execute(
                select(JOBS).where(
                    JOBS.c.job_kind == job_kind,
                    JOBS.c.auth_scope_hash == auth_context.token_hash,
                    JOBS.c.idempotency_key == idempotency_key,
                )
            ).mappings().first()
        if existing is None:
            raise
        existing_job = _row_to_dict(existing)
        if existing_job["request_hash"] != request_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Idempotency-Key was already used with a different request body",
            )
        return existing_job

    runtime.redis.rpush(_queue_name(runtime.settings, job_kind), row["id"])
    return row


def _update_job(runtime: Runtime, job_id: str, **fields: Any) -> dict[str, Any]:
    fields["updated_at"] = _utcnow()
    with _connection(runtime) as conn:
        conn.execute(update(JOBS).where(JOBS.c.id == job_id).values(**fields))
        row = conn.execute(select(JOBS).where(JOBS.c.id == job_id)).mappings().first()
    if row is None:
        raise RuntimeError(f"Job {job_id} disappeared during update")
    return _row_to_dict(row)


def _persist_artifacts(conn: Connection, job_id: str, artifacts: list[dict[str, Any]]) -> None:
    if not artifacts:
        return
    now = _utcnow()
    rows = [
        {
            "job_id": job_id,
            "role": artifact["role"],
            "uri": artifact["uri"],
            "content_type": artifact.get("content_type"),
            "metadata_json": artifact,
            "created_at": now,
        }
        for artifact in artifacts
    ]
    conn.execute(insert(JOB_ARTIFACTS), rows)


def _is_cancel_requested(engine: Engine, job_id: str) -> bool:
    with engine.begin() as conn:
        row = conn.execute(select(JOBS.c.cancel_requested).where(JOBS.c.id == job_id)).first()
    return bool(row[0]) if row else False


def _claim_job(runtime: Runtime, queue_name: str, worker_id: str) -> str | None:
    result = runtime.redis.blpop(queue_name, timeout=runtime.settings.queue_poll_timeout_seconds)
    if result is None:
        return None
    _, raw_job_id = result
    job_id = raw_job_id.decode("utf-8")
    now = _utcnow()
    lease_expires_at = datetime.fromtimestamp(
        now.timestamp() + runtime.settings.lease_ttl_seconds,
        tz=UTC,
    )
    with _connection(runtime) as conn:
        conn.execute(
            insert(WORKER_LEASES).values(
                worker_id=worker_id,
                queue_name=queue_name,
                job_id=job_id,
                heartbeat_at=now,
                lease_expires_at=lease_expires_at,
            )
        )
    return job_id


def _heartbeat_worker(runtime: Runtime, worker_id: str, queue_name: str, job_id: str | None) -> None:
    now = _utcnow()
    lease_expires_at = datetime.fromtimestamp(
        now.timestamp() + runtime.settings.lease_ttl_seconds,
        tz=UTC,
    )
    with _connection(runtime) as conn:
        conn.execute(
            update(WORKER_LEASES)
            .where(WORKER_LEASES.c.worker_id == worker_id)
            .values(
                queue_name=queue_name,
                job_id=job_id,
                heartbeat_at=now,
                lease_expires_at=lease_expires_at,
            )
        )


def _release_worker(runtime: Runtime, worker_id: str) -> None:
    with _connection(runtime) as conn:
        conn.execute(delete(WORKER_LEASES).where(WORKER_LEASES.c.worker_id == worker_id))


def _observe_terminal_job_metrics(
    job_kind: str,
    status: str,
    submitted_at: datetime | None,
    started_at: datetime | None,
    finished_at: datetime | None,
) -> None:
    if status not in TERMINAL_JOB_STATUSES:
        return
    observe_job_status(job_kind, status)
    if finished_at is None:
        return
    start_time = started_at or submitted_at or finished_at
    duration_s = max((finished_at - start_time).total_seconds(), 0.0)
    observe_job_duration(job_kind, status, duration_s)


def _process_job(runtime: Runtime, job_id: str, queue_name: str, worker_id: str) -> None:
    job = _fetch_job(runtime, job_id)
    if job["status"] == "cancelled":
        return
    if job["cancel_requested"] and job["status"] == "queued":
        updated = _update_job(runtime, job_id, status="cancelled", finished_at=_utcnow(), queue_position=None)
        _observe_terminal_job_metrics(
            updated["job_kind"],
            updated["status"],
            updated["submitted_at"],
            updated["started_at"],
            updated["finished_at"],
        )
        return
    job = _update_job(runtime, job_id, status="processing", started_at=_utcnow(), progress_pct=5)
    provider = _provider_from_job(runtime, job)
    try:
        _heartbeat_worker(runtime, worker_id, queue_name, job_id)
        result = provider.execute(runtime, job)
        finished_at = _utcnow() if result.status in TERMINAL_JOB_STATUSES else None
        result_json = dict(result.result)
        provider_metadata = dict(result.provider_metadata or {})
        if job["job_kind"] == "lora" and result.status == "completed":
            provider_metadata["trainer_backend"] = _resolve_trainer_backend(job, provider_metadata)
            if provider_metadata.get("dataset_manifest_hash") is None:
                LOG.warning("LoRA job %s completed without dataset_manifest_hash", job_id)
            if provider_metadata.get("weights_sha256") is None:
                LOG.warning("LoRA job %s completed without weights_sha256", job_id)
            if provider_metadata.get("config_sha256") is None:
                LOG.warning("LoRA job %s completed without config_sha256", job_id)
        weights_url = _coalesce(
            result_json.get("weights_url"),
            result_json.get("adapter_url"),
            result_json.get("model_url"),
        )
        if weights_url is not None:
            result_json["weights_url"] = weights_url
        with _connection(runtime) as conn:
            conn.execute(
                update(JOBS)
                .where(JOBS.c.id == job_id)
                .values(
                    status=result.status,
                    progress_pct=result.progress_pct,
                    provider_job_id=result.provider_job_id,
                    result_json=result_json,
                    provider_metadata_json=provider_metadata,
                    error_message=result.error_message,
                    queue_position=None,
                    finished_at=finished_at,
                    updated_at=_utcnow(),
                )
            )
            _persist_artifacts(conn, job_id, result.artifacts)
        _observe_terminal_job_metrics(
            job["job_kind"],
            result.status,
            job["submitted_at"],
            job["started_at"],
            finished_at,
        )
    except Exception as exc:
        LOG.exception("Worker failed processing job %s", job_id)
        updated = _update_job(
            runtime,
            job_id,
            status="dead_letter",
            progress_pct=0,
            error_message=str(exc),
            queue_position=None,
            finished_at=_utcnow(),
        )
        _observe_terminal_job_metrics(
            updated["job_kind"],
            updated["status"],
            updated["submitted_at"],
            updated["started_at"],
            updated["finished_at"],
        )


def _worker_loop(runtime: Runtime, job_kind: str, worker_index: int) -> None:
    queue_name = _queue_name(runtime.settings, job_kind)
    worker_id = f"{job_kind}-worker-{worker_index}"
    while not runtime.stop_event.is_set():
        try:
            job_id = _claim_job(runtime, queue_name, worker_id)
            if job_id is None:
                continue
            _process_job(runtime, job_id, queue_name, worker_id)
        except Exception:
            LOG.exception("Worker loop error for %s", worker_id)
            time.sleep(1)
        finally:
            _release_worker(runtime, worker_id)


def _sync_provider_capabilities(runtime: Runtime) -> None:
    now = _utcnow()
    with _connection(runtime) as conn:
        for provider in runtime.providers.values():
            payload = provider.capability.model_dump(mode="json")
            conn.execute(
                delete(PROVIDER_CAPABILITIES).where(
                    PROVIDER_CAPABILITIES.c.provider_key == provider.capability.provider_key
                )
            )
            conn.execute(
                insert(PROVIDER_CAPABILITIES).values(
                    provider_key=provider.capability.provider_key,
                    capabilities_json=payload,
                    refreshed_at=now,
                )
            )


def _requeue_unfinished_jobs(runtime: Runtime) -> None:
    for job_kind in ("lora", "chat", "image", "video"):
        runtime.redis.delete(_queue_name(runtime.settings, job_kind))
    with _connection(runtime) as conn:
        conn.execute(
            update(JOBS)
            .where(JOBS.c.status.in_(["queued", "processing"]))
            .values(status="queued", started_at=None, updated_at=_utcnow(), queue_position=None)
        )
        rows = conn.execute(select(JOBS).where(JOBS.c.status == "queued")).mappings().all()
    grouped: dict[str, list[dict[str, Any]]] = {"lora": [], "chat": [], "image": [], "video": []}
    for row in rows:
        grouped.setdefault(row["job_kind"], []).append(_row_to_dict(row))
    for job_kind, jobs in grouped.items():
        for index, job in enumerate(sorted(jobs, key=lambda item: item["created_at"]), start=1):
            _update_job(
                runtime,
                job["id"],
                queue_position=index,
                estimated_wait_seconds=_estimated_wait_seconds(runtime.settings, job_kind, index),
            )
            runtime.redis.rpush(_queue_name(runtime.settings, job_kind), job["id"])


def _initialize_runtime() -> Runtime:
    _validate_config(SETTINGS)
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.scratch_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(SETTINGS.database_url, future=True)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS exec"))
        METADATA.create_all(bind=conn)
    redis_client = Redis.from_url(SETTINGS.redis_url, decode_responses=False)
    storage = ArtifactStorage(SETTINGS)
    storage.ensure_bucket()
    runtime = Runtime(
        settings=SETTINGS,
        engine=engine,
        redis=redis_client,
        storage=storage,
        providers={
            "training.lora.local": LocalLoraTrainerProvider(),
            "chat.ollama": OllamaChatProvider(),
            "image.selfhosted": SelfHostedImageProvider(),
            "video.selfhosted": SelfHostedVideoProvider(),
        },
        stop_event=threading.Event(),
        threads=[],
    )
    _sync_provider_capabilities(runtime)
    _requeue_unfinished_jobs(runtime)
    for index in range(runtime.settings.lora_worker_count):
        thread = threading.Thread(target=_worker_loop, args=(runtime, "lora", index), daemon=True)
        runtime.threads.append(thread)
        thread.start()
    for index in range(runtime.settings.chat_worker_count):
        thread = threading.Thread(target=_worker_loop, args=(runtime, "chat", index), daemon=True)
        runtime.threads.append(thread)
        thread.start()
    for index in range(runtime.settings.image_worker_count):
        thread = threading.Thread(target=_worker_loop, args=(runtime, "image", index), daemon=True)
        runtime.threads.append(thread)
        thread.start()
    for index in range(runtime.settings.video_worker_count):
        thread = threading.Thread(target=_worker_loop, args=(runtime, "video", index), daemon=True)
        runtime.threads.append(thread)
        thread.start()
    return runtime


RUNTIME: Runtime | None = None


def _runtime() -> Runtime:
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = _initialize_runtime()
    return RUNTIME


def _shutdown_runtime() -> None:
    global RUNTIME
    if RUNTIME is None:
        return
    RUNTIME.stop_event.set()
    for thread in RUNTIME.threads:
        thread.join(timeout=5)
    RUNTIME.redis.close()
    RUNTIME.engine.dispose()
    RUNTIME = None


app = FastAPI(title="Self LoRA Service", version="0.2.0")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        path = request.url.path
        if path in {"/metrics", "/health/live"}:
            return await call_next(request)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            observe_request(request.method, path, status_code, time.perf_counter() - started)


app.add_middleware(MetricsMiddleware)
app.include_router(metrics_router)


@app.on_event("startup")
def startup() -> None:
    _runtime()


@app.on_event("shutdown")
def shutdown() -> None:
    _shutdown_runtime()


def _extract_auth_context(request: Request) -> AuthContext:
    settings = _runtime().settings
    if not settings.internal_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SELF_LORA_INTERNAL_TOKEN is not configured",
        )
    header_value = request.headers.get(settings.auth_header)
    if header_value is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    try:
        scheme, token = header_value.split(" ", 1)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized") from exc
    if scheme != settings.auth_scheme or not secrets.compare_digest(token, settings.internal_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return AuthContext(token_hash=_hash_text(token))


def _require_idempotency_key(idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> str:
    if not idempotency_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key header is required",
        )
    return idempotency_key


@app.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    runtime = _runtime()
    if not os.access(runtime.settings.scratch_dir, os.W_OK):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scratch dir not writable")
    with _connection(runtime) as conn:
        conn.execute(text("SELECT 1"))
    runtime.redis.ping()
    runtime.storage.ensure_bucket()
    return HealthResponse(status="ok")


@app.post("/v1/lora/jobs", response_model=AcceptedJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_lora_job(
    payload: CreateLoraJobRequest,
    request: Request,
    idempotency_key: str = Depends(_require_idempotency_key),
) -> AcceptedJobResponse:
    runtime = _runtime()
    auth_context = _extract_auth_context(request)
    job = _insert_job(
        runtime,
        job_kind="lora",
        job_type="train",
        provider_key="training.lora.local",
        auth_context=auth_context,
        idempotency_key=idempotency_key,
        payload=payload.model_dump(mode="json"),
    )
    observe_job_enqueued("lora", _job_tier(job["request_json"]), _queue_name(runtime.settings, "lora"))
    return _accepted_response(runtime, job)


@app.get("/v1/lora/jobs/{job_id}")
def get_lora_job(job_id: str, request: Request) -> dict[str, Any]:
    _extract_auth_context(request)
    runtime = _runtime()
    return _normalize_lora_result(_fetch_job(runtime, job_id))


@app.post("/v1/chat/jobs", response_model=AcceptedJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_chat_job(
    payload: CreateChatJobRequest,
    request: Request,
    idempotency_key: str = Depends(_require_idempotency_key),
) -> AcceptedJobResponse:
    runtime = _runtime()
    auth_context = _extract_auth_context(request)
    provider_key = payload.provider_hint or "chat.ollama"
    if provider_key not in runtime.providers:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown provider '{provider_key}'")
    job = _insert_job(
        runtime,
        job_kind="chat",
        job_type="completion",
        provider_key=provider_key,
        auth_context=auth_context,
        idempotency_key=idempotency_key,
        payload=payload.model_dump(mode="json"),
    )
    observe_job_enqueued("chat", _job_tier(job["request_json"]), _queue_name(runtime.settings, "chat"))
    return _accepted_response(runtime, job)


@app.get("/v1/chat/jobs/{job_id}")
def get_chat_job(job_id: str, request: Request) -> dict[str, Any]:
    _extract_auth_context(request)
    runtime = _runtime()
    return _normalize_chat_result(_fetch_job(runtime, job_id))


@app.post("/v1/image/jobs", response_model=AcceptedJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_image_job(
    payload: CreateImageJobRequest,
    request: Request,
    idempotency_key: str = Depends(_require_idempotency_key),
) -> AcceptedJobResponse:
    runtime = _runtime()
    auth_context = _extract_auth_context(request)
    job = _insert_job(
        runtime,
        job_kind="image",
        job_type="generate",
        provider_key="image.selfhosted",
        auth_context=auth_context,
        idempotency_key=idempotency_key,
        payload=payload.model_dump(mode="json"),
    )
    observe_job_enqueued("image", _job_tier(job["request_json"]), _queue_name(runtime.settings, "image"))
    return _accepted_response(runtime, job)


@app.get("/v1/image/jobs/{job_id}")
def get_image_job(job_id: str, request: Request) -> dict[str, Any]:
    _extract_auth_context(request)
    runtime = _runtime()
    return _normalize_image_result(_fetch_job(runtime, job_id))


@app.post("/v1/video/jobs", response_model=AcceptedJobResponse, status_code=status.HTTP_202_ACCEPTED)
def create_video_job(
    payload: CreateVideoJobRequest,
    request: Request,
    idempotency_key: str = Depends(_require_idempotency_key),
) -> AcceptedJobResponse:
    runtime = _runtime()
    auth_context = _extract_auth_context(request)
    job = _insert_job(
        runtime,
        job_kind="video",
        job_type="generate",
        provider_key="video.selfhosted",
        auth_context=auth_context,
        idempotency_key=idempotency_key,
        payload=payload.model_dump(mode="json"),
    )
    observe_job_enqueued("video", _job_tier(job["request_json"]), _queue_name(runtime.settings, "video"))
    return _accepted_response(runtime, job)


@app.get("/v1/video/jobs/{job_id}")
def get_video_job(job_id: str, request: Request) -> dict[str, Any]:
    _extract_auth_context(request)
    runtime = _runtime()
    return _normalize_video_result(_fetch_job(runtime, job_id))


@app.get("/v1/providers")
def list_providers(request: Request) -> dict[str, Any]:
    _extract_auth_context(request)
    runtime = _runtime()
    capabilities = [provider.capability.model_dump(mode="json") for provider in runtime.providers.values()]
    return {"providers": capabilities}


@app.post("/v1/jobs/{job_id}:cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    _extract_auth_context(request)
    runtime = _runtime()
    job = _fetch_job(runtime, job_id)
    if job["status"] in {"completed", "failed", "dead_letter", "cancelled"}:
        return {"id": job["id"], "status": job["status"]}
    updated = _update_job(runtime, job_id, cancel_requested=True)
    if updated["status"] == "queued":
        updated = _update_job(runtime, job_id, status="cancelled", finished_at=_utcnow())
        _observe_terminal_job_metrics(
            updated["job_kind"],
            updated["status"],
            updated["submitted_at"],
            updated["started_at"],
            updated["finished_at"],
        )
    return {"id": updated["id"], "status": updated["status"]}
