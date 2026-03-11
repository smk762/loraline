from __future__ import annotations

import re

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQ_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status_code"],
)

REQ_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)

JOB_ENQUEUED = Counter(
    "job_queue_enqueued_total",
    "Total jobs enqueued per queue tier.",
    ["job_kind", "tier", "queue_name"],
)

JOB_TRANSITIONS = Counter(
    "job_status_transitions_total",
    "Total terminal job status transitions.",
    ["job_kind", "status"],
)

JOB_DURATION = Histogram(
    "job_duration_seconds",
    "End-to-end job duration in seconds for terminal statuses.",
    ["job_kind", "status"],
    buckets=(5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 14400),
)

_UUID_SEGMENT_RE = re.compile(
    r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?=/|$)"
)
_INTEGER_SEGMENT_RE = re.compile(r"/\d+(?=/|$)")


def _normalize_path(path: str) -> str:
    normalized = _UUID_SEGMENT_RE.sub("/{id}", path)
    normalized = _INTEGER_SEGMENT_RE.sub("/{id}", normalized)
    return normalized


def observe_request(method: str, path: str, status_code: int, duration_s: float) -> None:
    normalized_path = _normalize_path(path)
    REQ_COUNT.labels(method=method, path=normalized_path, status_code=str(status_code)).inc()
    REQ_LATENCY.labels(method=method, path=normalized_path).observe(max(duration_s, 0.0))


def observe_job_enqueued(job_kind: str, tier: str, queue_name: str) -> None:
    JOB_ENQUEUED.labels(job_kind=job_kind, tier=tier, queue_name=queue_name).inc()


def observe_job_status(job_kind: str, status: str) -> None:
    JOB_TRANSITIONS.labels(job_kind=job_kind, status=status).inc()


def observe_job_duration(job_kind: str, status: str, duration_s: float) -> None:
    JOB_DURATION.labels(job_kind=job_kind, status=status).observe(max(duration_s, 0.0))


metrics_router = APIRouter()


@metrics_router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
