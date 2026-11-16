#!/usr/bin/env python3
"""Merge a trained LoRA/PEFT adapter back into the base init model, producing a
full standalone checkpoint that the merge-sweep eval harness
(evaluate_openjev_trait_probabilities.py + sweep_thresholds.py) can score with
no PEFT dependency. Output is a sharded bf16 safetensors model dir."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import openjev_compat  # noqa: F401  (restores BloomPreTrainedModel lost in transformers 5.16)

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer
from transformers import Qwen3_5ForConditionalGeneration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="merged init model dir")
    ap.add_argument("--adapter", type=Path, required=True, help="PEFT adapter dir")
    ap.add_argument("--output", type=Path, required=True, help="merged output dir")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # Load the base with the SAME model class the trainer used
    # (Qwen3_5ForConditionalGeneration), so the module graph/leaf naming matches the
    # saved adapter keys (linear_attn out_proj, in_proj_qkv/z/b/a, ...). Loading via
    # AutoModelForCausalLM produces a differently-named module tree for these hybrid
    # OpenJev mixing modules -> peft "missing adapter keys" -> merge applies nothing
    # -> the "merged" model equals the unmodified init (identical eval across methods).
    config = AutoConfig.from_pretrained(str(args.base), local_files_only=True)
    if config.model_type != "qwen3_5":
        raise ValueError("merge_peft_adapter requires a Qwen3.5 (qwen3_5) model")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.base),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)
    model.config.use_cache = True

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(args.output), safe_serialization=True, max_shard_size="5GB"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    tokenizer.save_pretrained(str(args.output))

    # carry over OpenJev runtime + template assets so the eval harness works
    for name in ("depth_config.json", "generation_config.json",
                 "chat_template.jinja",):
        src = args.base / name
        if src.exists():
            dst = args.output / name
            if not dst.exists():
                shutil.copy2(src, dst)
    rt = args.base / "openjet_runtime"
    if rt.is_dir():
        shutil.copytree(rt, args.output / "openjet_runtime", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (args.output / "merge_info.json").write_text(json.dumps({
        "merge_method": "peft_merge_and_unload",
        "base_model": str(args.base),
        "adapter": str(args.adapter),
        "output_dtype": "bfloat16",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"merged adapter -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
