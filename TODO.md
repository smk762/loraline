# TODO

## Current Status

- `self-lora` scaffold is running with:
  - PostgreSQL-backed job state
  - Redis queueing
  - MinIO/S3 artifact uploads
  - wrapper toolkit for image/video/chat/voice
  - Cloudflare Tunnel overlay
- Connectivity check succeeded for:
  - PostgreSQL
  - Redis
  - S3/MinIO
  - Ollama
  - `tss-stack`

## Next Priority

- Implement a real image LoRA backend wrapper behind `SELF_LORA_IMAGE_LORA_BACKEND_COMMAND`
- Implement a real video LoRA backend wrapper behind `SELF_LORA_VIDEO_LORA_BACKEND_COMMAND`
- Add preview generation for real image/video training outputs
- Add provider-specific metadata and artifact manifests for trained models

## Training Backends

- Add `trainers/kohya_wrapper.py` or equivalent for image LoRA
- Add a video-training wrapper once the local video stack/GPU path is ready
- Decide which base models are supported per backend and document compatibility
- Add richer result metadata:
  - model family
  - adapter format
  - training hyperparameters
  - dataset fingerprints

## Inference / Modality Follow-Up

- Decide whether chat stays owned by `agent-composer` or moves behind a dedicated internal proxy
- Keep voice on `tss-stack` unless voice-adapter training is added later
- Add future image/video inference endpoints only when local generation services are available

## Operations

- Decide whether to run:
  - LAN only
  - Cloudflare Tunnel only
  - LAN + Cloudflare together
- Add host firewall allowlists for `self-lora`
- Consider adding CI checks for:
  - Python syntax
  - connectivity checker
  - container build

## Helpful Files

- `README.md`
- `APP_CONTRACT.md`
- `docs/cloudflare-zero-trust.md`
- `docs/agent-composer-networking.md`
- `trainers/router.py`
- `scripts/check_connections.py`
