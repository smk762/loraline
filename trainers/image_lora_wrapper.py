from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from common import JobSpec, ProviderResult, parse_job_args, run_backend_command, write_result


def main() -> int:
    args = parse_job_args("Image LoRA backend wrapper.")
    job_spec = JobSpec.from_path(args.job_spec)
    backend_mode = os.getenv("SELF_LORA_IMAGE_LORA_BACKEND_MODE", "command").strip().lower() or "command"
    command_text = os.getenv("SELF_LORA_IMAGE_LORA_BACKEND_COMMAND", "").strip()

    if backend_mode != "command":
        result = ProviderResult(
            status="dead_letter",
            progress_pct=0,
            error_message=f"Unsupported SELF_LORA_IMAGE_LORA_BACKEND_MODE '{backend_mode}'",
            metadata={"wrapper": "image_lora_wrapper", "backend_mode": backend_mode},
        )
        write_result(args.result_path, result)
        return 1

    if not command_text:
        result = ProviderResult(
            status="dead_letter",
            progress_pct=0,
            error_message="SELF_LORA_IMAGE_LORA_BACKEND_COMMAND is not configured",
            metadata={"wrapper": "image_lora_wrapper", "backend_mode": backend_mode},
        )
        write_result(args.result_path, result)
        return 1

    completed = run_backend_command(
        command_text,
        ["--job-spec", str(args.job_spec), "--result-path", str(args.result_path)],
    )
    if completed.returncode != 0:
        result = ProviderResult(
            status="dead_letter",
            progress_pct=0,
            error_message=completed.stderr.strip() or completed.stdout.strip() or "Image trainer failed",
            metadata={"wrapper": "image_lora_wrapper", "backend_mode": backend_mode},
        )
        write_result(args.result_path, result)
        return 1

    if not args.result_path.exists():
        result = ProviderResult(
            status="dead_letter",
            progress_pct=0,
            error_message="Image trainer completed without writing a result manifest",
            metadata={"wrapper": "image_lora_wrapper", "backend_mode": backend_mode},
        )
        write_result(args.result_path, result)
        return 1

    payload = json.loads(args.result_path.read_text(encoding="utf-8"))
    payload.setdefault("metadata", {})
    payload["metadata"].setdefault("wrapper", "image_lora_wrapper")
    payload["metadata"].setdefault("backend_mode", backend_mode)
    payload["metadata"].setdefault("base_model", job_spec.base_model)
    args.result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
