#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def env_value(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]
        if key in values and values[key] != "":
            return values[key]
    return ""


def socket_probe(host: str, port: int, timeout_seconds: float) -> None:
    with socket.create_connection((host, port), timeout=timeout_seconds):
        return


def http_probe(url: str, timeout_seconds: float) -> tuple[int, str]:
    request = Request(url, headers={"User-Agent": "self-lora-connectivity-check/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read(256).decode("utf-8", errors="ignore").strip()
        return response.status, body


def parse_host_port_from_url(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"Could not determine hostname from '{url}'")
    return parsed.hostname, parsed.port or default_port


def describe_exception(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return f"{exc.__class__.__name__}: HTTP {exc.code}"
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            return f"{exc.__class__.__name__}: timeout"
        return f"{exc.__class__.__name__}: {reason}"
    if isinstance(exc, socket.timeout):
        return f"{exc.__class__.__name__}: timeout"
    return f"{exc.__class__.__name__}: {exc}"


def check_postgres(values: dict[str, str], timeout_seconds: float) -> CheckResult:
    database_url = env_value(values, "SELF_LORA_DATABASE_URL", "DATABASE_URL")
    if not database_url:
        return CheckResult("postgres", "SKIPPED", "DATABASE_URL is not configured")
    try:
        import psycopg

        connect_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(connect_url, connect_timeout=int(timeout_seconds)) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return CheckResult("postgres", "ACTIVE", f"{database_url} authenticated query succeeded")
    except Exception as exc:
        try:
            host, port = parse_host_port_from_url(database_url, 5432)
            socket_probe(host, port, timeout_seconds)
            return CheckResult(
                "postgres",
                "PARTIAL",
                f"{database_url} port reachable but auth/query failed: {describe_exception(exc)}",
            )
        except Exception:
            return CheckResult("postgres", "INACTIVE", f"{database_url} {describe_exception(exc)}")


def check_redis(values: dict[str, str], timeout_seconds: float) -> CheckResult:
    redis_url = env_value(values, "SELF_LORA_REDIS_URL", "REDIS_URL")
    if not redis_url:
        return CheckResult("redis", "SKIPPED", "REDIS_URL is not configured")
    try:
        from redis import Redis

        client = Redis.from_url(redis_url, socket_connect_timeout=timeout_seconds, socket_timeout=timeout_seconds)
        client.ping()
        client.close()
        return CheckResult("redis", "ACTIVE", f"{redis_url} PING succeeded")
    except Exception as exc:
        try:
            host, port = parse_host_port_from_url(redis_url, 6379)
            socket_probe(host, port, timeout_seconds)
            return CheckResult(
                "redis",
                "PARTIAL",
                f"{redis_url} port reachable but PING failed: {describe_exception(exc)}",
            )
        except Exception:
            return CheckResult("redis", "INACTIVE", f"{redis_url} {describe_exception(exc)}")


def check_s3(values: dict[str, str], timeout_seconds: float) -> CheckResult:
    endpoint_url = env_value(values, "SELF_LORA_S3_ENDPOINT_URL", "S3_ENDPOINT_URL")
    bucket = env_value(values, "SELF_LORA_S3_BUCKET", "S3_BUCKET")
    access_key = env_value(values, "SELF_LORA_S3_ACCESS_KEY_ID", "S3_ACCESS_KEY_ID")
    secret_key = env_value(values, "SELF_LORA_S3_SECRET_ACCESS_KEY", "S3_SECRET_ACCESS_KEY")
    region = env_value(values, "SELF_LORA_S3_REGION", "S3_REGION") or "us-east-1"
    if not endpoint_url:
        return CheckResult("s3", "SKIPPED", "S3 endpoint is not configured")
    try:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        if bucket:
            client.head_bucket(Bucket=bucket)
            return CheckResult("s3", "ACTIVE", f"{endpoint_url} bucket '{bucket}' is reachable")
        client.list_buckets()
        return CheckResult("s3", "ACTIVE", f"{endpoint_url} endpoint is reachable")
    except Exception as exc:
        try:
            status_code, _ = http_probe(endpoint_url, timeout_seconds)
            return CheckResult(
                "s3",
                "PARTIAL",
                f"{endpoint_url} HTTP {status_code} but S3 auth/bucket failed: {describe_exception(exc)}",
            )
        except Exception:
            return CheckResult("s3", "INACTIVE", f"{endpoint_url} {describe_exception(exc)}")


def check_ollama(values: dict[str, str], timeout_seconds: float) -> CheckResult:
    base_url = env_value(values, "SELF_LORA_OLLAMA_BASE_URL")
    if not base_url:
        return CheckResult("ollama", "SKIPPED", "SELF_LORA_OLLAMA_BASE_URL is not configured")
    target_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        status_code, body = http_probe(target_url, timeout_seconds)
        return CheckResult("ollama", "ACTIVE", f"{target_url} HTTP {status_code} {body[:120]}")
    except Exception as exc:
        try:
            host, port = parse_host_port_from_url(base_url, 11434)
            socket_probe(host, port, timeout_seconds)
            return CheckResult(
                "ollama",
                "PARTIAL",
                f"{target_url} port reachable but HTTP probe failed: {describe_exception(exc)}",
            )
        except Exception:
            return CheckResult("ollama", "INACTIVE", f"{target_url} {describe_exception(exc)}")


def check_tss(values: dict[str, str], timeout_seconds: float) -> CheckResult:
    base_url = env_value(values, "SELF_LORA_TSS_BASE_URL")
    if not base_url:
        return CheckResult("tss-stack", "SKIPPED", "SELF_LORA_TSS_BASE_URL is not configured")
    for path in ("/health", "/v1/capabilities", "/voices"):
        target_url = f"{base_url.rstrip('/')}{path}"
        try:
            status_code, body = http_probe(target_url, timeout_seconds)
            return CheckResult("tss-stack", "ACTIVE", f"{target_url} HTTP {status_code} {body[:120]}")
        except Exception as exc:
            continue
    try:
        host, port = parse_host_port_from_url(base_url, 9001)
        socket_probe(host, port, timeout_seconds)
        return CheckResult(
            "tss-stack",
            "PARTIAL",
            f"{base_url} port reachable but no known endpoint responded",
        )
    except Exception as exc:
        return CheckResult("tss-stack", "INACTIVE", f"{base_url} {describe_exception(exc)}")


def render_results(results: list[CheckResult]) -> None:
    width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.name.ljust(width)}  {result.status:<8}  {result.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check configured service connectivity from .env.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH, help="Path to env file to inspect.")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-check timeout in seconds.")
    args = parser.parse_args(argv)

    env_values = load_env_file(args.env_file)
    checks: list[Callable[[dict[str, str], float], CheckResult]] = [
        check_postgres,
        check_redis,
        check_s3,
        check_ollama,
        check_tss,
    ]
    results = [check(env_values, args.timeout) for check in checks]
    render_results(results)

    statuses = {result.status for result in results}
    if "INACTIVE" in statuses:
        return 1
    if "PARTIAL" in statuses:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
