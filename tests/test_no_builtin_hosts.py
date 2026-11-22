"""The Ollama and tss-stack base URLs have no built-in host: unset, a job dead-letters with the variable's name."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRAINERS = Path(__file__).resolve().parents[1] / "trainers"


class UnsetBaseUrlTest(unittest.TestCase):
    def run_wrapper(self, name: str, env_name: str) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            request, result = Path(tmp) / "request.json", Path(tmp) / "result.json"
            request.write_text(json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "body": {}}))
            env = {k: v for k, v in os.environ.items() if k != env_name}
            proc = subprocess.run(
                [sys.executable, name, "--request-path", str(request), "--result-path", str(result)],
                cwd=TRAINERS, env=env, capture_output=True, text=True, timeout=60, check=False,
            )
            return proc.returncode, json.loads(result.read_text())

    def test_each_wrapper_dead_letters_without_its_base_url(self) -> None:
        for name, env_name in [
            ("chat_ollama_wrapper.py", "SELF_LORA_OLLAMA_BASE_URL"),
            ("voice_tts_wrapper.py", "SELF_LORA_TSS_BASE_URL"),
            ("voice_stt_wrapper.py", "SELF_LORA_TSS_BASE_URL"),
        ]:
            with self.subTest(name):
                code, result = self.run_wrapper(name, env_name)
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "dead_letter")
                self.assertEqual(result["error_message"], f"{env_name} is not configured")


if __name__ == "__main__":
    unittest.main()
