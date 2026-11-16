#!/usr/bin/env python3
"""DPO / RLCD fine-tune of a merged OpenJev clickbait init, reusing quality_jev's
binary-candidate scaffold (openjev_training.compile_record / build_training_records).

Every row already has input_ids + candidate_ids=[not_clickbait, clickbait] + target,
so a preference pair is chosen = gold candidate token vs rejected = the wrong one.
A single OpenJev early-exit forward on the [1,2] candidate set yields logits over
BOTH candidates, so we take log-probs from that one pass -- no double forward.

  * --method dpo : Direct Preference Optimization on that token preference, with
    a frozen reference model (the untouched init) as anchor (beta).
  * --method rlcd: RL from Contrastive Demonstrations. Minimize CE to the CHOSEN
    completion while maximizing divergence from the REJECTED one.

Runs single-pass B=1 through the OpenJev early-exit executor exactly like
finetune_openjev_clickbait.py. Emits the best adapter by high-exit eval F1.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")

import openjev_compat  # noqa: F401  (restores BloomPreTrainedModel lost in transformers 5.16)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from openjev_training import (
    binary_metrics,
    build_training_records,
    compile_record,
)


DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--ref-model-dir", type=Path, required=True,
                    help="frozen anchor (DPO reference / RLCD baseline)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--method", choices=("dpo", "rlcd"), required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO regularization / RLCD contrast strength")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-train-rows", type=int)
    ap.add_argument("--max-eval-rows", type=int)
    return ap.parse_args()


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = local_rank = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OpenJev preference training")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return rows if limit is None else rows[:limit]


def load_runtime(model_dir: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(model_dir))
    try:
        contracts = importlib.import_module("openjet_runtime.contracts")
        early_exit = importlib.import_module("openjet_runtime.early_exit")
    finally:
        sys.path.pop(0)
    return contracts.render_prompt, contracts.label_mapping, early_exit.QwenEarlyExit


def build_items(rows, tokenizer, render_prompt, label_mapping, max_length):
    return [
        compile_record(rec, tokenizer, render_prompt=render_prompt,
                       label_mapping=label_mapping, max_length=max_length)
        for rec in build_training_records(rows, "direct")
    ]


class OpenJevPrefModel(torch.nn.Module):
    """High-exit only wrapper. One forward on the [1,2] candidate set returns
    logits over both candidates; we slice chosen/rejected log-probs from it."""

    def __init__(self, model, wrapper_type, *, high_depth):
        super().__init__()
        self.executor = wrapper_type(model)
        self.high_depth = high_depth

    def forward(self, input_ids, candidate_ids):
        decisions = self.executor(input_ids, candidate_ids, depths=(self.high_depth,))
        return decisions[0].logits  # [1, 2] logits over [not_clickbait, clickbait]


def chosen_rejected_logprobs(logits, target):
    # logits: [1, 2] over [not_clickbait, clickbait]; target = gold index
    logp = torch.log_softmax(logits.float(), dim=-1).squeeze(0)  # [2]
    return logp[target], logp[1 - target]


def dpo_loss(logp_ch, logp_rej, ref_ch, ref_rej, beta):
    # DPO in log-prob space: -log sigmoid(beta * ((lcp-lrp) - (lcref-lrref)))
    implicit = (logp_ch - logp_rej) - (ref_ch - ref_rej)
    return -torch.log(torch.sigmoid(beta * implicit))


def rlcd_loss(logp_ch, logp_rej, beta):
    # CE toward chosen (gold) + contrastive push away from rejected
    chosen_ce = -logp_ch
    contrast = -torch.log((1.0 - torch.exp(logp_rej)).clamp_min(1e-12))
    return chosen_ce + beta * contrast


def collate_items(items):
    if len(items) != 1:
        raise ValueError("OpenJev early-exit executor requires --batch-size 1")
    item = items[0]
    return {
        "input_ids": torch.tensor([item["input_ids"]], dtype=torch.long),
        "candidate_ids": torch.tensor(item["candidate_ids"], dtype=torch.long),
        "targets": torch.tensor([item["target"]], dtype=torch.long),
        "meta": item,
    }


@torch.no_grad()
def evaluate(model, items, device):
    model.eval()
    labels, probs = [], []
    for item in items:
        batch = collate_items([item])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["input_ids"].to(device),
                           batch["candidate_ids"].to(device))
        target = int(batch["targets"].item())
        labels.append(target)
        probs.append(float(torch.softmax(logits.float(), -1)[0, 1].cpu()))
    return {"high": binary_metrics(labels, probs)}


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size must be 1 for the audited Q1 executor")
    rank, local_rank, world_size = distributed_context()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda", local_rank)

    model_dir = args.model_dir.expanduser().resolve()
    ref_dir = args.ref_model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    depth_config = json.loads((model_dir / "depth_config.json").read_text("utf-8"))
    high_depth = int(depth_config["full_depth"])
    render_prompt, label_mapping, wrapper_type = load_runtime(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    train_items = build_items(
        load_rows(args.data_dir / "train.jsonl", args.max_train_rows),
        tokenizer, render_prompt, label_mapping, args.max_length)
    eval_items = build_items(
        load_rows(args.data_dir / "eval.jsonl", args.max_eval_rows),
        tokenizer, render_prompt, label_mapping, args.max_length)
    if not train_items or not eval_items:
        raise ValueError("train and eval splits must both be nonempty")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise ValueError("preference trainer requires Qwen3.5 conditional generation")

    def load_policy(path, lora_config, trainable):
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa")
        base.config.use_cache = False
        if lora_config is not None:
            base = get_peft_model(base, lora_config)
        if not trainable:
            for p in base.parameters():
                p.requires_grad_(False)
        return base.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(DEFAULT_LORA_TARGETS), bias="none")

    policy = load_policy(model_dir, lora_config, trainable=True)
    ref = load_policy(ref_dir, None, trainable=False)

    model = OpenJevPrefModel(policy, wrapper_type, high_depth=high_depth)
    ref_model = OpenJevPrefModel(ref, wrapper_type, high_depth=high_depth)
    train_model = DDP(model, device_ids=[local_rank]) if world_size > 1 else model

    trainable = [p for p in train_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    sampler = (DistributedSampler(train_items, num_replicas=world_size, rank=rank,
                                  shuffle=True, drop_last=False)
               if world_size > 1 else None)
    loader = DataLoader(train_items, batch_size=1,
                        shuffle=sampler is None, sampler=sampler,
                        collate_fn=collate_items, num_workers=0, pin_memory=True)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(steps_per_epoch * args.epochs * args.warmup_ratio),
        num_training_steps=steps_per_epoch * args.epochs)

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_dir = output_dir / "best"
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_seen = 0
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            candidate_ids = batch["candidate_ids"].to(device)
            target = int(batch["targets"].item())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = train_model(input_ids, candidate_ids)
                ch, rej = chosen_rejected_logprobs(logits, target)
                if args.method == "dpo":
                    with (torch.no_grad()):
                        rlogits = ref_model(input_ids, candidate_ids)
                        rch, rrej = chosen_rejected_logprobs(rlogits, target)
                    loss = dpo_loss(ch, rej, rch, rrej, args.beta)
                else:
                    loss = rlcd_loss(ch, rej, args.beta)
                scaled = loss / args.gradient_accumulation_steps
            scaled.backward()
            total_loss += float(loss.detach())
            n_seen += 1
            if (step % args.gradient_accumulation_steps == 0
                    or step == len(loader)):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            eval_metrics = evaluate(model, eval_items, device)
            epoch_result = {"epoch": epoch + 1,
                            "train_loss": total_loss / max(n_seen, 1),
                            "eval": eval_metrics}
            history.append(epoch_result)
            print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
            sel = eval_metrics["high"]["f1"]
            if sel > best_f1:
                best_f1 = sel
                best_epoch = epoch + 1
                best_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(best_dir, safe_serialization=True)
                tokenizer.save_pretrained(best_dir)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "training_report.json").write_text(
            json.dumps({
                "method": args.method,
                "base_model": str(model_dir),
                "ref_model": str(ref_dir),
                "beta": args.beta,
                "train_rows": len(train_items),
                "eval_rows": len(eval_items),
                "best_epoch": best_epoch,
                "best_eval_f1": best_f1,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "lora_rank": args.lora_rank,
                "history": history,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"method": args.method, "best_eval_f1": best_f1,
                          "best_epoch": best_epoch}, ensure_ascii=False), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
