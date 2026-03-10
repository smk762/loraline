from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    user_id: str
    companion_id: str | None
    name: str
    base_model: str
    trigger_word: str
    training_steps: int
    dataset_dir: Path
    dataset_urls: list[str]
    output_dir: Path
    metadata: dict[str, Any]

    @classmethod
    def from_path(cls, path: Path) -> "JobSpec":
        payload = read_json(path)
        return cls(
            job_id=payload["job_id"],
            user_id=payload["user_id"],
            companion_id=payload.get("companion_id"),
            name=payload["name"],
            base_model=payload["base_model"],
            trigger_word=payload["trigger_word"],
            training_steps=int(payload["training_steps"]),
            dataset_dir=Path(payload["dataset_dir"]),
            dataset_urls=list(payload.get("dataset_urls") or []),
            output_dir=Path(payload["output_dir"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ProviderResult:
    status: str
    progress_pct: int
    provider_job_id: str | None = None
    weights_path: str | None = None
    adapter_path: str | None = None
    model_path: str | None = None
    preview_path: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "progress_pct": self.progress_pct,
            "provider_job_id": self.provider_job_id,
            "weights_path": self.weights_path,
            "adapter_path": self.adapter_path,
            "model_path": self.model_path,
            "preview_path": self.preview_path,
            "error_message": self.error_message,
            "metadata": self.metadata or {},
            "result": self.result or {},
        }
        return {key: value for key, value in payload.items() if value is not None}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_result(path: Path, result: ProviderResult) -> None:
    write_json(path, result.to_dict())


def parse_job_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser.parse_args()


def parse_request_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--request-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser.parse_args()


def run_backend_command(command_text: str, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    command = shlex.split(command_text) + extra_args
    return subprocess.run(command, check=False, capture_output=True, text=True)


def poll_async_job(
    client: httpx.Client,
    status_url: str,
    *,
    terminal_statuses: set[str],
    poll_interval_seconds: float = 1.0,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    started_at = time.monotonic()
    while True:
        response = client.get(status_url)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") in terminal_statuses:
            return payload
        if time.monotonic() - started_at > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for async job at {status_url}")
        time.sleep(poll_interval_seconds)
