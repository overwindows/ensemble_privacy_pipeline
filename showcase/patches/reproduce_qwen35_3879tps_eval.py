#!/usr/bin/env python3
"""
Standalone Reproducible Online HTTP (/v1/chat/completions) Benchmark Harness
for Qwen3.5-4B (maiprofile-qwen35-sft) on 1x NVIDIA A100-80GB (vLLM 0.31.0).

Reproduces:
  1. Wave-6g Champion (4-Patch Stack, Greedy T=0.0): 3,879.50 output tok/s
     (magenta_book_0g0hp7t2cy | +92.66 tok/s / +2.45% vs r9 | 12,728.44 total tok/s | 3.01 QPS | 663.54s | 2000/2000 JSON)
  2. Wave-6g 3-Patch Stack (Greedy T=0.0):           3,877.46 output tok/s
     (frank_jelly_p13140lkhv  | +90.62 tok/s / +2.39% vs r9 | 12,741.56 total tok/s | 3.02 QPS | 662.41s | 2000/2000 JSON)
  3. Wave-6f Stock Control (0 Patches, Greedy T=0.0): 3,843.12 output tok/s
     (hungry_cat_jqbhqqvz0z   | +56.28 tok/s / +1.49% vs r9 | 12,606.40 total tok/s | 2.98 QPS | 670.03s | 1998/2000 JSON)
  4. Wave-6e Champion Stack (Stochastic T=0.2):       3,827.28 output tok/s
     (ivory_box_09wc1wnqzl    | +40.44 tok/s / +1.07% vs r9 | 12,519.51 total tok/s | 2.96 QPS | 675.51s | 1998/2000 JSON)

Usage against a running vLLM server:
  python3 reproduce_qwen35_3879tps_eval.py \\
      --base-url http://127.0.0.1:8100 \\
      --data-path /path/to/l1_v2_gpt56_test.jsonl \\
      --output-dir ./repro_results

Usage that also spawns serve_qwen35_v031_optimized.sh automatically:
  python3 reproduce_qwen35_3879tps_eval.py \\
      --launch-server ./serve_qwen35_v031_optimized.sh \\
      --model-path /path/to/MAI-Profile-Qwen35-SFT-checkpoint \\
      --data-path /path/to/l1_v2_gpt56_test.jsonl \\
      --output-dir ./repro_results
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import aiohttp

EXPECTED_REQUEST_COUNT = 2000
EXPECTED_PROMPT_TOKENS = 5871663
EXPECTED_SERVED_MODEL_NAME = "maiprofile-qwen35-sft"
EXPECTED_MAX_TOKENS = 8192
METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[+-]?Inf|NaN)$"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Qwen3.5-4B 3,879.50 output tok/s online HTTP /v1/chat/completions benchmark."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8100", help="vLLM OpenAI HTTP server base URL")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to l1_v2_gpt56_test.jsonl (2,000 rows)")
    parser.add_argument("--output-dir", type=Path, default=Path("./repro_results"), help="Output directory for JSON/metrics")
    parser.add_argument("--served-model-name", default=EXPECTED_SERVED_MODEL_NAME, help="Served model name")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature override (default: 0.0 for strict greedy; pass -1 to preserve dataset row temperature=0.2)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top_p override (default: 1.0)",
    )
    parser.add_argument("--max-tokens", type=int, default=EXPECTED_MAX_TOKENS, help="Max output tokens per request (default: 8192)")
    parser.add_argument("--concurrency", type=int, default=2000, help="Client HTTP concurrency (default: 2000)")
    parser.add_argument("--warmup-requests", type=int, default=8, help="Pre-timer HTTP warmup requests (default: 8)")
    parser.add_argument("--warmup-max-tokens", type=int, default=16, help="Max output tokens for warmup requests (default: 16)")
    parser.add_argument("--expected-mtp-depth", type=int, default=3, help="Expected MTP draft tokens per step (default: 3)")
    parser.add_argument("--request-timeout-s", type=float, default=3600.0, help="Per-request HTTP timeout seconds")
    parser.add_argument("--ready-timeout-s", type=float, default=900.0, help="Server readiness timeout seconds")
    parser.add_argument(
        "--launch-server",
        type=Path,
        default=None,
        help="Optional path to serve_qwen35_v031_optimized.sh to launch and manage automatically",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Model checkpoint directory when using --launch-server",
    )
    parser.add_argument("--port", type=int, default=8100, help="Port when using --launch-server")
    return parser.parse_args()


def load_requests(
    data_path: Path,
    *,
    served_model_name: str,
    temperature_override: float,
    top_p_override: float,
    max_tokens_override: int,
) -> list[dict[str, Any]]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Benchmark dataset not found: {data_path}")

    requests: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Line {line_number} has missing/empty messages")

            req: dict[str, Any] = {
                "model": served_model_name,
                "messages": messages,
                "max_tokens": int(max_tokens_override or payload.get("max_tokens", EXPECTED_MAX_TOKENS)),
                "temperature": float(payload.get("temperature", 0.0)) if temperature_override < 0 else float(temperature_override),
                "top_p": float(top_p_override),
            }
            requests.append(req)

    if len(requests) != EXPECTED_REQUEST_COUNT:
        print(
            f"[warn] Dataset row count is {len(requests)} (expected {EXPECTED_REQUEST_COUNT} for full-2000 benchmark)",
            file=sys.stderr,
        )
    return requests


def wait_for_server(base_url: str, served_model_name: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "server not checked"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as health_resp:
                if health_resp.status != 200:
                    last_error = f"/health status={health_resp.status}"
                    time.sleep(2)
                    continue
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as models_resp:
                models_payload = json.loads(models_resp.read().decode("utf-8"))
            model_ids = [item.get("id") for item in models_payload.get("data", []) if isinstance(item, dict)]
            if served_model_name in model_ids:
                return models_payload
            last_error = f"served model {served_model_name!r} not in {model_ids!r}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"vLLM server at {base_url} did not become ready within {timeout_s}s: {last_error}")


def scrape_metrics_text(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_prometheus_metrics(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples: list[tuple[str, dict[str, str], float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = {k: v for k, v in LABEL_RE.findall(raw_labels or "")}
        samples.append((name, labels, value))
    return samples


def sum_metric(samples: list[tuple[str, dict[str, str], float]], metric_name: str, **required_labels: str) -> float:
    total = 0.0
    for name, labels, value in samples:
        if name != metric_name:
            continue
        if all(labels.get(k) == v for k, v in required_labels.items()):
            total += value
    return total


def mtp_counter_delta(before_text: str, after_text: str) -> dict[str, Any]:
    before = parse_prometheus_metrics(before_text)
    after = parse_prometheus_metrics(after_text)

    def delta(metric_name: str, **labels: str) -> float:
        return sum_metric(after, metric_name, **labels) - sum_metric(before, metric_name, **labels)

    num_drafts = delta("vllm:spec_decode_num_drafts_total")
    if num_drafts == 0.0:
        num_drafts = delta("vllm:spec_decode_num_drafts")
    num_draft_tokens = delta("vllm:spec_decode_num_draft_tokens_total")
    if num_draft_tokens == 0.0:
        num_draft_tokens = delta("vllm:spec_decode_num_draft_tokens")
    num_accepted_tokens = delta("vllm:spec_decode_num_accepted_tokens_total")
    if num_accepted_tokens == 0.0:
        num_accepted_tokens = delta("vllm:spec_decode_num_accepted_tokens")

    per_pos_names = {
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        "vllm:spec_decode_num_accepted_tokens_per_pos",
    }
    positions = sorted(
        {
            labels["position"]
            for name, labels, _ in after
            if name in per_pos_names and "position" in labels
        },
        key=lambda item: int(item) if item.isdigit() else item,
    )
    accepted_per_position: dict[str, float] = {}
    for pos in positions:
        pos_delta = delta("vllm:spec_decode_num_accepted_tokens_per_pos_total", position=pos)
        if pos_delta == 0.0:
            pos_delta = delta("vllm:spec_decode_num_accepted_tokens_per_pos", position=pos)
        accepted_per_position[pos] = pos_delta

    acceptance_rate = (num_accepted_tokens / num_draft_tokens) if num_draft_tokens > 0 else 0.0
    mean_acceptance_length = (1.0 + (num_accepted_tokens / num_drafts)) if num_drafts > 0 else 0.0
    mean_drafted_per_event = (num_draft_tokens / num_drafts) if num_drafts > 0 else 0.0
    per_pos_rates = {
        pos: (count / num_drafts if num_drafts > 0 else 0.0)
        for pos, count in accepted_per_position.items()
    }
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "accepted_tokens_per_position": accepted_per_position,
        "draft_acceptance_rate": acceptance_rate,
        "mean_acceptance_length_including_bonus": mean_acceptance_length,
        "mean_drafted_tokens_per_event": mean_drafted_per_event,
        "per_position_acceptance_rates": per_pos_rates,
        "engagement_verified": num_drafts > 0 and num_draft_tokens > 0 and num_accepted_tokens > 0,
    }


async def run_warmup(
    base_url: str,
    requests: list[dict[str, Any]],
    warmup_requests: int,
    warmup_max_tokens: int,
) -> None:
    if warmup_requests <= 0 or not requests:
        return
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=warmup_requests, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for idx in range(min(warmup_requests, len(requests))):
            payload = dict(requests[idx])
            payload["max_tokens"] = warmup_max_tokens
            tasks.append(
                session.post(f"{base_url}/v1/chat/completions", json=payload)
            )
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for resp in responses:
            if isinstance(resp, aiohttp.ClientResponse):
                await resp.read()
                resp.release()
    print(f"[warmup] Completed {min(warmup_requests, len(requests))} warmup requests (max_tokens={warmup_max_tokens}).", flush=True)


async def send_one_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    idx: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        try:
            async with session.post(url, json=payload) as response:
                raw_body = await response.text()
                latency_s = time.perf_counter() - started
                if response.status != 200:
                    return {
                        "index": idx,
                        "ok": False,
                        "status": response.status,
                        "latency_s": latency_s,
                        "error": raw_body[:1000],
                    }
                data = json.loads(raw_body)
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                reasoning_content = message.get("reasoning_content")
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                json_valid = False
                if isinstance(content, str) and content.strip():
                    try:
                        json.loads(content)
                        json_valid = True
                    except Exception:  # noqa: BLE001
                        json_valid = False
                return {
                    "index": idx,
                    "ok": True,
                    "status": response.status,
                    "latency_s": latency_s,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
                    "finish_reason": choice.get("finish_reason"),
                    "json_parseable": json_valid,
                    "has_reasoning_content": reasoning_content not in (None, ""),
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "index": idx,
                "ok": False,
                "status": None,
                "latency_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def evaluate_all(
    base_url: str,
    requests: list[dict[str, Any]],
    concurrency: int,
    request_timeout_s: float,
) -> tuple[list[dict[str, Any]], float]:
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    semaphore = asyncio.Semaphore(concurrency)
    url = f"{base_url}/v1/chat/completions"

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(send_one_request(session, semaphore, url, idx, payload))
            for idx, payload in enumerate(requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_seconds = time.perf_counter() - t0
    results.sort(key=lambda item: int(item["index"]))
    return results, wall_seconds


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")

    server_proc: subprocess.Popen[str] | None = None
    try:
        if args.launch_server is not None:
            if args.model_path is None:
                raise ValueError("--model-path is required when using --launch-server")
            server_log = args.output_dir / "vllm_server.log"
            cmd = ["bash", str(args.launch_server), str(args.port), "1", str(args.model_path)]
            print(f"[server] Launching: {' '.join(cmd)} (log -> {server_log})", flush=True)
            log_handle = server_log.open("w", encoding="utf-8")
            server_proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            base_url = f"http://127.0.0.1:{args.port}"

        requests = load_requests(
            args.data_path,
            served_model_name=args.served_model_name,
            temperature_override=args.temperature,
            top_p_override=args.top_p,
            max_tokens_override=args.max_tokens,
        )
        effective_temps = sorted({float(r.get("temperature", 0.0)) for r in requests})
        print(
            f"[init] Loaded {len(requests)} requests from {args.data_path} "
            f"(effective temperatures={effective_temps}, top_p={args.top_p}, max_tokens={args.max_tokens})",
            flush=True,
        )

        wait_for_server(base_url, args.served_model_name, args.ready_timeout_s)
        print(f"[ready] vLLM server at {base_url} is healthy and serving {args.served_model_name!r}.", flush=True)

        asyncio.run(run_warmup(base_url, requests, args.warmup_requests, args.warmup_max_tokens))

        metrics_before = scrape_metrics_text(base_url)
        (args.output_dir / "metrics_before.txt").write_text(metrics_before, encoding="utf-8")

        print(f"[eval] Dispatching {len(requests)} concurrent /v1/chat/completions requests...", flush=True)
        records, wall_seconds = asyncio.run(
            evaluate_all(base_url, requests, args.concurrency, args.request_timeout_s)
        )

        metrics_after = scrape_metrics_text(base_url)
        (args.output_dir / "metrics_after.txt").write_text(metrics_after, encoding="utf-8")

        successes = [r for r in records if r.get("ok")]
        failures = [r for r in records if not r.get("ok")]
        prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in successes)
        output_tokens = sum(int(r.get("completion_tokens", 0)) for r in successes)
        json_valid_count = sum(1 for r in successes if r.get("json_parseable"))
        mtp_stats = mtp_counter_delta(metrics_before, metrics_after)

        output_tps = output_tokens / wall_seconds if wall_seconds > 0 else 0.0
        total_tps = (prompt_tokens + output_tokens) / wall_seconds if wall_seconds > 0 else 0.0
        qps = len(successes) / wall_seconds if wall_seconds > 0 else 0.0

        summary = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "http_online_chat_completions",
            "dataset": str(args.data_path),
            "requests": len(records),
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "wall_seconds": wall_seconds,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "qps": qps,
            "output_tps": output_tps,
            "total_tps": total_tps,
            "mean_input_tokens": (prompt_tokens / len(successes)) if successes else 0.0,
            "mean_output_tokens": (output_tokens / len(successes)) if successes else 0.0,
            "json_parseable": json_valid_count,
            "json_parseable_fraction_of_successes": (json_valid_count / len(successes)) if successes else 0.0,
            "effective_temperatures": effective_temps,
            "greedy_decoding_verified": effective_temps == [0.0],
            "mtp_counters": mtp_stats,
            "mtp_engagement_verified": bool(mtp_stats["engagement_verified"]),
        }

        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

        if len(failures) > 0:
            print(f"[error] {len(failures)} requests failed!", file=sys.stderr)
            return 2
        if len(records) == EXPECTED_REQUEST_COUNT and prompt_tokens != EXPECTED_PROMPT_TOKENS:
            print(
                f"[warn] Prompt token count {prompt_tokens} != canonical {EXPECTED_PROMPT_TOKENS} "
                "(check --default-chat-template-kwargs '{\"enable_thinking\": false}')",
                file=sys.stderr,
            )
        return 0
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
