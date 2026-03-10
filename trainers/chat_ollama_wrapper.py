from __future__ import annotations

import os

import httpx

from common import parse_request_args, read_json, write_json


def main() -> int:
    args = parse_request_args("Ollama chat wrapper.")
    request_payload = read_json(args.request_path)
    base_url = os.getenv("SELF_LORA_OLLAMA_BASE_URL", "http://192.168.1.109:11434").rstrip("/")
    timeout_seconds = float(os.getenv("SELF_LORA_OLLAMA_TIMEOUT_SECONDS", "120"))

    body = {
        "model": request_payload["model"],
        "messages": request_payload["messages"],
        "stream": False,
        "options": request_payload.get("options") or {},
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(f"{base_url}/api/chat", json=body)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        write_json(
            args.result_path,
            {
                "status": "dead_letter",
                "error_message": str(exc),
                "metadata": {"wrapper": "chat_ollama_wrapper", "base_url": base_url},
            },
        )
        return 1

    result = {
        "status": "completed",
        "model": payload.get("model", request_payload["model"]),
        "message": payload.get("message"),
        "content": (payload.get("message") or {}).get("content"),
        "done_reason": payload.get("done_reason"),
        "total_duration": payload.get("total_duration"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "metadata": {"wrapper": "chat_ollama_wrapper", "base_url": base_url},
    }
    write_json(args.result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
