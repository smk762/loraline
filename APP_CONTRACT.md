## App Contract

This file defines exactly what the main app expects from the self-hosted LoRA
service. It is intended to move with this folder into its own repo.

### Purpose

The main app owns:

- auth
- tenants/users
- companion state and memory
- billing and gem accounting
- LoRA job records in PostgreSQL
- websocket fan-out to clients

The self-hosted LoRA service owns:

- GPU scheduling
- dataset download/preparation
- model fine-tuning
- preview generation
- artifact upload or publishing

The LoRA service must stay stateless with respect to product data. It must not
be the source of truth for user accounts, companion memory, billing, or access
control policy.

### Required Environment Assumptions

The app calls the service using these settings:

- `LORA_BACKEND=selfhosted` or any non-`mock` value
- `LORA_API_BASE_URL`
- `LORA_API_KEY`
- `LORA_API_TIMEOUT_MS`
- `LORA_JOB_TIMEOUT_MS`
- `LORA_JOB_POLL_INTERVAL_MS`
- `LORA_API_AUTH_HEADER`
- `LORA_API_AUTH_SCHEME`
- `LORA_API_EXTRA_HEADERS_JSON`
- `LORA_API_JOBS_PATH`

Default path expected by the app:

```text
/v1/lora/jobs
```

Default auth header expected by the app:

```text
Authorization: Bearer <LORA_API_KEY>
```

Idempotency behavior expected by the app:

```text
Idempotency-Key: <opaque-client-generated-value>
```

Rules:

- The service should require `Idempotency-Key` on submit endpoints.
- Replaying the same key with the same normalized request should return the
  original accepted job response.
- Replaying the same key with a different request body should return `409`.

### Required API

#### 1. Create training job

```http
POST /v1/lora/jobs
Authorization: Bearer <token>
Idempotency-Key: <value>
Content-Type: application/json
```

Request body sent by the app:

```json
{
  "user_id": "uuid-or-app-user-id",
  "name": "My Custom LoRA",
  "base_model": "flux_dev",
  "trigger_word": "ohwx person",
  "dataset_urls": [
    "https://cdn.example.com/uploads/img1.png"
  ],
  "training_steps": 1000,
  "metadata": {
    "description": "optional free text"
  }
}
```

Required behavior:

- Validate bearer token.
- Validate `base_model`.
- Accept app-owned dataset URLs.
- Create a durable job record on the LoRA side.
- Return a stable provider-side job ID immediately.
- Return `202 Accepted` on success.

Accepted response contract:

```json
{
  "id": "provider-job-id",
  "status": "queued",
  "estimated_duration_minutes": 15,
  "cost_gems": 0,
  "estimated_wait_seconds": 5,
  "queue_position": 1,
  "status_url": "http://lora-host:8010/v1/lora/jobs/provider-job-id"
}
```

Minimum required fields:

- `id`
- `status`

Additional fields the app understands when present:

- `estimated_duration_minutes`
- `cost_gems`
- `estimated_wait_seconds`
- `queue_position`
- `status_url`

### 2. Get training job status

```http
GET /v1/lora/jobs/{job_id}
Authorization: Bearer <token>
```

Response must be a JSON object. The app polls this endpoint until terminal state.

Required fields:

- `id`
- `status`

Recognized terminal statuses:

- `completed`
- `failed`
- `dead_letter`
- `cancelled`

Recognized non-terminal statuses:

- `queued`
- `processing`

Completed response example:

```json
{
  "id": "provider-job-id",
  "status": "completed",
  "progress_pct": 100,
  "preview_url": "https://cdn.example.com/loras/previews/abc.png",
  "weights_url": "s3://bucket/loras/user-1/model.safetensors",
  "updated_at": "2026-03-09T12:24:00Z"
}
```

Failed response example:

```json
{
  "id": "provider-job-id",
  "status": "failed",
  "progress_pct": 0,
  "error_message": "CUDA out of memory"
}
```

The app reads these optional fields when present:

- `preview_url`
- `weights_url`
- `adapter_url`
- `model_url`
- `error_message`
- `progress_pct`
- `result.preview_url`
- `result.weights_url`

If you prefer nesting final outputs, this also works:

```json
{
  "id": "provider-job-id",
  "status": "completed",
  "result": {
    "preview_url": "https://cdn.example.com/loras/previews/abc.png",
    "weights_url": "s3://bucket/loras/user-1/model.safetensors"
  }
}
```

Alias handling required by the app:

- If the provider produces `adapter_url` or `model_url`, the service should
  normalize those to `weights_url` in the top-level response where possible.
- `error_message` is the preferred failure field for LoRA jobs.

### 3. Optional Internal APIs

These are useful for operations and future provider routing, but they are not
required by the current app-side LoRA poller:

```http
GET /v1/providers
POST /v1/jobs/{job_id}:cancel
POST /v1/chat/jobs
GET /v1/chat/jobs/{job_id}
```

Notes:

- The core app owns websocket fan-out to clients over `/v1/ws`.
- Direct provider SSE is optional and not required for the current MVP.
- Callbacks/webhooks are optional future capability, not current scope.

### Operational Expectations

- The service must be private to the LAN/VPN or reverse-proxied privately.
- Training requests must be idempotent at the infrastructure level where possible.
- Job records must survive container restarts.
- Large datasets should be streamed or downloaded to disk, not held fully in RAM.
- The service should cap concurrent training jobs.
- The service should reject unsupported base models explicitly.
- The service should expose `/health/live` and `/health/ready`.

### Artifact Expectations

- `dataset_urls` are owned by the main app.
- Datasets should be streamed/downloaded to scratch disk, not buffered in RAM.
- Final weights should be published to durable storage.
- `weights_url` should be stable and reusable by downstream generation services.
- `preview_url` should point to a preview image suitable for app/UI display.
- If object storage is used, stable object keys should be tenant- and
  job-scoped.

### What The App Stores

The app stores these LoRA job fields in PostgreSQL:

- local app `id`
- `user_id`
- `name`
- `base_model`
- `trigger_word`
- `training_images_json`
- `training_steps`
- `description`
- `status`
- `provider_job_id`
- `preview_url`
- `weights_url`
- `cost_gems`
- `estimated_duration_minutes`
- `error_message`

Important:

- The app's LoRA job ID is not the same as the provider's job ID.
- The service must treat `user_id` only as tenancy metadata from the trusted app.
- Companion memory/personality state remains app-side and is not managed here.

### Execution Service Environment

The current scaffold expects these service-side settings:

- `SELF_LORA_INTERNAL_TOKEN`
- `SELF_LORA_AUTH_HEADER`
- `SELF_LORA_AUTH_SCHEME`
- `SELF_LORA_DATABASE_URL`
- `SELF_LORA_REDIS_URL`
- `SELF_LORA_S3_ENDPOINT_URL`
- `SELF_LORA_S3_ACCESS_KEY_ID`
- `SELF_LORA_S3_SECRET_ACCESS_KEY`
- `SELF_LORA_S3_BUCKET`
- `SELF_LORA_S3_REGION`
- `SELF_LORA_CDN_BASE_URL`
- `SELF_LORA_OLLAMA_BASE_URL`
- `SELF_LORA_TRAINER_COMMAND`
- `SELF_LORA_TRAINER_PROVIDER`
- `SELF_LORA_IMAGE_LORA_COMMAND`
- `SELF_LORA_IMAGE_LORA_BACKEND_MODE`
- `SELF_LORA_IMAGE_LORA_BACKEND_COMMAND`
- `SELF_LORA_VIDEO_LORA_COMMAND`
- `SELF_LORA_VIDEO_LORA_BACKEND_MODE`
- `SELF_LORA_VIDEO_LORA_BACKEND_COMMAND`
- `IMAGE_BACKEND`
- `IMAGE_API_BASE_URL`
- `IMAGE_API_KEY`
- `IMAGE_API_GENERATE_PATH`
- `IMAGE_API_IMG2IMG_PATH`
- `IMAGE_API_JOBS_PATH`
- `IMAGE_API_TIMEOUT_MS`
- `IMAGE_API_JOB_TIMEOUT_MS`
- `IMAGE_API_JOB_POLL_INTERVAL_MS`
- `IMAGE_API_AUTH_HEADER`
- `IMAGE_API_AUTH_SCHEME`
- `VIDEO_BACKEND`
- `VIDEO_API_BASE_URL`
- `VIDEO_API_KEY`
- `VIDEO_API_GENERATE_PATH`
- `VIDEO_API_JOBS_PATH`
- `VIDEO_API_TIMEOUT_MS`
- `VIDEO_API_JOB_TIMEOUT_MS`
- `VIDEO_API_JOB_POLL_INTERVAL_MS`
- `VIDEO_API_AUTH_HEADER`
- `VIDEO_API_AUTH_SCHEME`
- `SELF_LORA_CHAT_WRAPPER_COMMAND`
- `SELF_LORA_TTS_WRAPPER_COMMAND`
- `SELF_LORA_STT_WRAPPER_COMMAND`
- `SELF_LORA_TSS_BASE_URL`
- `SELF_LORA_TSS_TIMEOUT_SECONDS`
- `SELF_LORA_TSS_POLL_SECONDS`

### Local Trainer Subprocess Contract

The local LoRA provider currently executes an external trainer command:

```text
SELF_LORA_TRAINER_COMMAND --job-spec <path> --result-path <path>
```

The execution service writes a JSON job spec containing:

- `job_id`
- `user_id`
- `companion_id`
- `name`
- `base_model`
- `trigger_word`
- `training_steps`
- `dataset_dir`
- `dataset_urls`
- `output_dir`
- `metadata`

The trainer process must write a JSON result manifest to `--result-path`.
Supported result fields:

- `status`
- `progress_pct`
- `provider_job_id`
- `weights_path`
- `adapter_path`
- `model_path`
- `preview_path`
- `error_message`
- `metadata`
- `result`

Any output file paths should point to local files that the execution service can
upload to object storage before the final poll response is returned.

### Recommended Router Path

For a multi-user, multi-provider host, prefer a stable router entrypoint:

```text
SELF_LORA_TRAINER_COMMAND=python3 /srv/self-lora/trainers/router.py
SELF_LORA_TRAINER_PROVIDER=mock-image-lora
```

Available scaffold providers today:

- `mock-image-lora`: writes placeholder weights and preview artifacts for smoke tests.
- `external-command`: backward-compatible alias that delegates to `SELF_LORA_IMAGE_LORA_COMMAND`.
- `external-image-command`: delegates image LoRA jobs to `SELF_LORA_IMAGE_LORA_COMMAND`.
- `external-video-command`: delegates video LoRA jobs to `SELF_LORA_VIDEO_LORA_COMMAND`.

Example real-trainer delegation:

```text
SELF_LORA_TRAINER_PROVIDER=external-image-command
SELF_LORA_IMAGE_LORA_COMMAND=python3 /opt/self-lora/trainers/kohya_wrapper.py
```

This keeps the execution-service contract stable while allowing different image,
video, or remote-compute training backends to be swapped in later.

Recommended modality hint for LoRA requests:

```json
{
  "metadata": {
    "modality": "video"
  }
}
```

### Supporting Wrapper Toolkit

The scaffold now includes supporting wrappers for each modality:

- `trainers/image_lora_wrapper.py`
- `trainers/video_lora_wrapper.py`
- `trainers/chat_ollama_wrapper.py`
- `trainers/voice_tts_wrapper.py`
- `trainers/voice_stt_wrapper.py`

Usage patterns:

- Image/video LoRA wrappers use `--job-spec <path> --result-path <path>`.
- Chat/voice wrappers use `--request-path <path> --result-path <path>`.

Recommended environment wiring:

```text
SELF_LORA_IMAGE_LORA_COMMAND=python3 /srv/self-lora/trainers/image_lora_wrapper.py
SELF_LORA_IMAGE_LORA_BACKEND_MODE=command
SELF_LORA_IMAGE_LORA_BACKEND_COMMAND=python3 /opt/self-lora/trainers/kohya_wrapper.py
SELF_LORA_VIDEO_LORA_COMMAND=python3 /srv/self-lora/trainers/video_lora_wrapper.py
SELF_LORA_VIDEO_LORA_BACKEND_MODE=command
SELF_LORA_VIDEO_LORA_BACKEND_COMMAND=python3 /opt/self-lora/trainers/video_lora_backend.py
SELF_LORA_CHAT_WRAPPER_COMMAND=python3 /srv/self-lora/trainers/chat_ollama_wrapper.py
SELF_LORA_TTS_WRAPPER_COMMAND=python3 /srv/self-lora/trainers/voice_tts_wrapper.py
SELF_LORA_STT_WRAPPER_COMMAND=python3 /srv/self-lora/trainers/voice_stt_wrapper.py
```

Notes:

- `chat_ollama_wrapper.py` is useful if chat execution is later moved out of the
  API process into its own worker.
- `voice_tts_wrapper.py` and `voice_stt_wrapper.py` bridge to the existing
  `tss-stack` async API and poll until terminal state.
- The current service implementation directly handles chat in-process, but these
  wrappers make the modality boundaries explicit for future queue workers.
