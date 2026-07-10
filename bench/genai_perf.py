"""NVIDIA GenAI-Perf / AIPerf benchmark wrapper for local Qwen3-8B inference.

This module can either:
  1. Build the GenAI-Perf / AIPerf CLI command for an existing OpenAI-compatible endpoint, or
  2. Start a minimal local HTTP server that serves the local Qwen3-8B implementation and then
     invoke GenAI-Perf/AIPerf against it.

Examples:
  python bench/genai_perf.py --serve-local --weights weights --client genai-perf \
      --input-tokens 256 --output-tokens 128 --num-prompts 32

  python bench/genai_perf.py --endpoint http://127.0.0.1:8000/v1/chat/completions \
      --client aiperf --input-tokens 256 --output-tokens 128 --num-prompts 32
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from baseline import load_from_hf_dir
    from tokenizer import Qwen3Tokenizer
except Exception:  # pragma: no cover - environment-dependent
    load_from_hf_dir = None  # type: ignore[assignment]
    Qwen3Tokenizer = None  # type: ignore[assignment]


def build_genai_perf_command(
    executable: str,
    endpoint: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_prompts: int,
    concurrency: int = 1,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the CLI command for NVIDIA GenAI-Perf or AIPerf."""
    cmd = [
        executable,
        "profile",
        "--service-kind",
        "openai",
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--input-tokens-mean",
        str(input_tokens),
        "--output-tokens-mean",
        str(output_tokens),
        "--num-prompts",
        str(num_prompts),
        "--concurrency",
        str(concurrency),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for local inference benchmarking. Install torch and rerun.")


class _LocalInferenceHandler(BaseHTTPRequestHandler):
    server_version = "qwen3mma/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = {"status": "ok"}
            self._send_json(payload, HTTPStatus.OK)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        _require_torch()
        if self.path not in {"/v1/chat/completions", "/v1/completions"}:
            self._send_json({"error": "unsupported endpoint"}, HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        data = json.loads(body.decode("utf-8") or "{}")

        messages = data.get("messages")
        if not messages:
            prompt = data.get("prompt") or data.get("input") or "Hello"
            messages = [{"role": "user", "content": prompt}]

        input_ids = self.server.tokenizer.apply_chat_template(messages)
        input_tensor = torch.tensor([input_ids], device=self.server.device)

        with torch.inference_mode():
            output_ids = self.server.model.generate(
                input_tensor,
                max_new_tokens=self.server.max_new_tokens,
                temperature=self.server.temperature,
                top_p=self.server.top_p,
                eos_token_id=Qwen3Tokenizer.EOS_ID,
            )

        new_text = self.server.tokenizer.decode(output_ids[0, len(input_ids):].tolist())
        response = {
            "id": f"cmpl-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.server.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": new_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": max(0, len(output_ids[0]) - len(input_ids)),
                "total_tokens": len(output_ids[0]),
            },
        }
        self._send_json(response, HTTPStatus.OK)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _LocalInferenceServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[_LocalInferenceHandler],
        model: torch.nn.Module,
        tokenizer: Qwen3Tokenizer,
        device: torch.device,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p


def _wait_for_server(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            import urllib.request

            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as resp:
                if resp.getcode() == HTTPStatus.OK:
                    return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for local benchmark server at http://{host}:{port}")


def _load_local_model(weights_dir: str | Path, device: str, dtype: str):
    _require_torch()
    assert load_from_hf_dir is not None and Qwen3Tokenizer is not None
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}. Run download_weights.py first.")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    model = load_from_hf_dir(str(weights_path), device=device)
    model = model.to(torch_dtype).eval()
    tokenizer = Qwen3Tokenizer.from_dir(str(weights_path))
    return model, tokenizer


def serve_local_benchmark(
    weights_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
    dtype: str = "bfloat16",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[threading.Thread, _LocalInferenceServer]:
    _require_torch()
    model, tokenizer = _load_local_model(weights_dir, device, dtype)
    server = _LocalInferenceServer(
        (host, port),
        _LocalInferenceHandler,
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        model_name="qwen3-8b",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _wait_for_server(host, port)
    return thread, server


def run_benchmark(
    endpoint: str,
    client: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_prompts: int,
    concurrency: int = 1,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    command = build_genai_perf_command(
        executable=client,
        endpoint=endpoint,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_prompts=num_prompts,
        concurrency=concurrency,
        extra_args=extra_args,
    )
    print("Running:", " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return None
    return subprocess.run(command, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=["genai-perf", "aiperf"], default="genai-perf")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible endpoint to benchmark")
    parser.add_argument("--serve-local", action="store_true", help="Start a local HTTP server and benchmark it")
    parser.add_argument("--weights", default="weights", help="Path to the downloaded HuggingFace weights")
    parser.add_argument("--model", default="qwen3-8b", help="Model identifier passed to GenAI-Perf")
    parser.add_argument("--input-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--device",
        default="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoint = args.endpoint
    if args.serve_local:
        if not endpoint:
            endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"
        if args.dry_run:
            print(f"Dry-run: would start local Qwen3-8B benchmark server on {endpoint}")
            return run_benchmark(
                endpoint=endpoint,
                client=args.client,
                model=args.model,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                num_prompts=args.num_prompts,
                concurrency=args.concurrency,
                dry_run=True,
            ) is None and 0 or 0
        print(f"Starting local Qwen3-8B benchmark server on {endpoint} ...")
        thread, server = serve_local_benchmark(
            weights_dir=args.weights,
            host=args.host,
            port=args.port,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        try:
            result = run_benchmark(
                endpoint=endpoint,
                client=args.client,
                model=args.model,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                num_prompts=args.num_prompts,
                concurrency=args.concurrency,
                dry_run=args.dry_run,
            )
            if result is not None:
                return result.returncode
            return 0
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    if not endpoint:
        raise SystemExit("Provide --endpoint or use --serve-local")

    result = run_benchmark(
        endpoint=endpoint,
        client=args.client,
        model=args.model,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        num_prompts=args.num_prompts,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )
    return 0 if result is None else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
