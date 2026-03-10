## Self-LoRA VM Bundle

This folder contains a minimal deployment bundle for a dedicated GPU host
on your LAN.

What is included:

- `compose.yaml`: local VM stack using `docker compose`
- `compose.cloudflare.yaml`: optional Cloudflare Tunnel overlay
- `Dockerfile`: container image for the placeholder control-plane API
- `requirements.txt`: Python dependencies
- `service.py`: minimal private API for LoRA job intake and status lookup
- `APP_CONTRACT.md`: exact interface the main app expects from this service
- `.env.example`: required environment variables
- `docs/cloudflare-zero-trust.md`: tunnel and Access deployment notes
- `docs/agent-composer-networking.md`: internal/LAN networking guidance
- `systemd/self-lora.service`: boot-time service wrapper

What this bundle does today:

- exposes private job endpoints over HTTP
- enforces an internal bearer token
- stores durable job state in PostgreSQL
- uses Redis queues and worker threads for async job processing
- uploads training artifacts to S3/MinIO
- can route chat jobs to external Ollama
- includes a router-based trainer entrypoint for local LoRA backends
- provides health and readiness endpoints

What you still need to add on the GPU host:

- a real image/video LoRA trainer behind `SELF_LORA_IMAGE_LORA_BACKEND_COMMAND` and `SELF_LORA_VIDEO_LORA_BACKEND_COMMAND`
- optional inference endpoints if the host will also serve LoRA-backed generation
- GPU-aware scheduling if one GPU must handle both training and inference

Recommended host layout:

- install at `/opt/self-lora`
- persist state under `/var/lib/self-lora`
- keep datasets, weights, and previews in S3/MinIO when possible
- restrict network access to your app/workers LAN only

Quick start:

```bash
cp .env.example .env
docker compose up -d --build
curl http://<vm-ip>:8010/health/live
```

CI:

- GitHub Actions runs Python syntax validation on `service.py`, `scripts/`, and `trainers/`
- GitHub Actions performs a Docker build smoke test for the production image
- GitHub Actions runs Trivy filesystem and container image scans for high and critical issues

API capabilities:

- `POST /v1/lora/jobs` and `GET /v1/lora/jobs/{job_id}` handle LoRA training jobs
- `POST /v1/image/jobs` and `GET /v1/image/jobs/{job_id}` proxy async image generation jobs to the `imogen` gateway
- `POST /v1/video/jobs` and `GET /v1/video/jobs/{job_id}` proxy async video generation jobs to the `vidita` gateway
- LoRA training modality routing uses `metadata.modality=image|video` and dispatches through `trainers/router.py`

Connectivity check:

```bash
python3 scripts/check_connections.py
```

Optional flags:

```bash
python3 scripts/check_connections.py --env-file .env --timeout 5
```

Status meanings:

- `ACTIVE`: the service responded successfully using the configured credentials or API.
- `PARTIAL`: the host/port is reachable, but the full authenticated check failed.
- `INACTIVE`: the check could not connect to the target.
- `SKIPPED`: the required env var was not configured.

Cloudflare Zero Trust overlay:

```bash
docker compose -f compose.yaml -f compose.cloudflare.yaml up -d --build
```

Notes:

- This overlay removes the normal host port binding and publishes through `cloudflared`.
- Add `CLOUDFLARED_TUNNEL_TOKEN` to `.env` before using it.
- See `docs/cloudflare-zero-trust.md` for the tunnel-only and LAN-plus-tunnel options.
- Self-hosted image/video gateway env settings are included in `.env.example` and passed through `compose.yaml`.

Agent-composer internal/LAN guidance:

- See `docs/agent-composer-networking.md`
- Recommended pattern: keep `ollama` internal-only, expose only the app/reverse proxy, and firewall `self-lora` to trusted source IPs.

Systemd:

```bash
sudo cp systemd/self-lora.service /etc/systemd/system/self-lora.service
sudo systemctl daemon-reload
sudo systemctl enable --now self-lora
```

Current trainer routing notes:

1. Keep `SELF_LORA_TRAINER_COMMAND="python3 /srv/self-lora/trainers/router.py"` as the stable entrypoint.
2. Use request `metadata.modality=image|video` to select the correct LoRA wrapper.
3. Set `SELF_LORA_IMAGE_LORA_BACKEND_COMMAND` and `SELF_LORA_VIDEO_LORA_BACKEND_COMMAND` to real training backends when they are available.

Trainer router examples:

```bash
# smoke-test artifacts
SELF_LORA_TRAINER_COMMAND="python3 /srv/self-lora/trainers/router.py"
SELF_LORA_TRAINER_PROVIDER=mock-image-lora

# delegate to a real wrapper
SELF_LORA_TRAINER_COMMAND="python3 /srv/self-lora/trainers/router.py"
SELF_LORA_TRAINER_PROVIDER=external-image-command
SELF_LORA_IMAGE_LORA_COMMAND="python3 /srv/self-lora/trainers/image_lora_wrapper.py"
SELF_LORA_IMAGE_LORA_BACKEND_COMMAND="python3 /opt/self-lora/trainers/kohya_wrapper.py"

# video LoRA routing
SELF_LORA_TRAINER_COMMAND="python3 /srv/self-lora/trainers/router.py"
SELF_LORA_VIDEO_LORA_COMMAND="python3 /srv/self-lora/trainers/video_lora_wrapper.py"
SELF_LORA_VIDEO_LORA_BACKEND_COMMAND="python3 /opt/self-lora/trainers/video_lora_backend.py"
```

Wrapper toolkit:

- `trainers/router.py`: stable top-level training router
- `trainers/image_lora_wrapper.py`: image LoRA adapter wrapper
- `trainers/video_lora_wrapper.py`: video LoRA adapter wrapper
- `trainers/chat_ollama_wrapper.py`: chat request wrapper for Ollama
- `trainers/voice_tts_wrapper.py`: async TTS bridge for `tss-stack`
- `trainers/voice_stt_wrapper.py`: async STT bridge for `tss-stack`

Typical commands:

```bash
# image LoRA
python3 /srv/self-lora/trainers/image_lora_wrapper.py --job-spec /tmp/job-spec.json --result-path /tmp/result.json

# video LoRA
python3 /srv/self-lora/trainers/video_lora_wrapper.py --job-spec /tmp/job-spec.json --result-path /tmp/result.json

# chat
python3 /srv/self-lora/trainers/chat_ollama_wrapper.py --request-path /tmp/chat-request.json --result-path /tmp/chat-result.json

# voice tts
python3 /srv/self-lora/trainers/voice_tts_wrapper.py --request-path /tmp/tts-request.json --result-path /tmp/tts-result.json

# voice stt
python3 /srv/self-lora/trainers/voice_stt_wrapper.py --request-path /tmp/stt-request.json --result-path /tmp/stt-result.json
```

Example chat request payload:

```json
{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "Hello there" }
  ],
  "options": {
    "temperature": 0.7
  }
}
```

Example TTS request payload:

```json
{
  "body": {
    "text": "Hello from async voice",
    "voice_id": "aria-en",
    "format": "mp3"
  }
}
```

Example STT request payload:

```json
{
  "body": {
    "audio_url": "https://cdn.example.com/audio/input.mp3",
    "language": "en-US"
  }
}
```
