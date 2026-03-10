from __future__ import annotations

import os
import uuid

import httpx

from common import parse_request_args, poll_async_job, read_json, write_json


def main() -> int:
    args = parse_request_args("Voice TTS wrapper for tss-stack.")
    request_payload = read_json(args.request_path)
    base_url = os.getenv("SELF_LORA_TSS_BASE_URL", "http://192.168.1.138:9001").rstrip("/")
    timeout_seconds = float(os.getenv("SELF_LORA_TSS_TIMEOUT_SECONDS", "300"))
    poll_seconds = float(os.getenv("SELF_LORA_TSS_POLL_SECONDS", "1.5"))
    headers = {"Idempotency-Key": request_payload.get("idempotency_key") or str(uuid.uuid4())}

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            submit = client.post(f"{base_url}/v1/tts", json=request_payload["body"], headers=headers)
            if submit.status_code == 404:
                submit = client.post(f"{base_url}/tts/jobs", json=request_payload["body"], headers=headers)
            submit.raise_for_status()
            accepted = submit.json()
            job_id = accepted["id"]
            status_url = request_payload.get("status_url") or f"{base_url}/tts/jobs/{job_id}"
            final_payload = poll_async_job(
                client,
                status_url,
                terminal_statuses={"completed", "failed", "dead_letter", "cancelled"},
                poll_interval_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )
    except Exception as exc:
        write_json(
            args.result_path,
            {
                "status": "dead_letter",
                "error_message": str(exc),
                "metadata": {"wrapper": "voice_tts_wrapper", "base_url": base_url},
            },
        )
        return 1

    write_json(
        args.result_path,
        {
            "status": final_payload.get("status"),
            "job_id": final_payload.get("id"),
            "audio_url": final_payload.get("audio_url"),
            "format": final_payload.get("format"),
            "duration_seconds": final_payload.get("duration_seconds"),
            "voice_id": final_payload.get("voice_id"),
            "progress_pct": final_payload.get("progress_pct"),
            "error_message": final_payload.get("error_message"),
            "metadata": {"wrapper": "voice_tts_wrapper", "base_url": base_url},
        },
    )
    return 0 if final_payload.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
