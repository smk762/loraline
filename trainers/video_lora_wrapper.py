from __future__ import annotations

import json
import os

from common import JobSpec, ProviderResult, parse_job_args, run_backend_command, write_result


def main() -> int:
    args = parse_job_args("Video LoRA backend wrapper.")
    job_spec = JobSpec.from_path(args.job_spec)
    command_text = os.getenv("SELF_LORA_VIDEO_LORA_BACKEND_COMMAND", "").strip()

    if not command_text:
        result = ProviderResult(
            status="dead_letter",
            progress_pct=0,
            error_message="SELF_LORA_VIDEO_LORA_BACKEND_COMMAND is not configured",
            metadata={
                "wrapper": "video_lora_wrapper",
                "note": "Configure a real video training backend when GPU/video services are ready.",
            },
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
            error_message=completed.stderr.strip() or completed.stdout.strip() or "Video trainer failed",
            metadata={"wrapper": "video_lora_wrapper"},
        )
        write_result(args.result_path, result)
        return 1

    if not args.result_path.exists():
        result = ProviderResult(
            status="dead_letter",
            progress_pct=0,
            error_message="Video trainer completed without writing a result manifest",
            metadata={"wrapper": "video_lora_wrapper"},
        )
        write_result(args.result_path, result)
        return 1

    payload = json.loads(args.result_path.read_text(encoding="utf-8"))
    payload.setdefault("metadata", {})
    payload["metadata"].setdefault("wrapper", "video_lora_wrapper")
    payload["metadata"].setdefault("base_model", job_spec.base_model)
    args.result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
