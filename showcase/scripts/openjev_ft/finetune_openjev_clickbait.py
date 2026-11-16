#!/usr/bin/env python3
"""Task-specific approximation of the published OpenJev two-exit training."""

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
    LOSS_MODES,
    TASK_MODES,
    LossConfig,
    binary_metrics,
    build_training_records,
    compile_record,
    compute_training_loss,
    fit_temperature,
)


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a merged OpenJev/Qwen3.5 checkpoint with the published "
            "joint-depth objective or an optional Laya-style proper-scoring loss."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="openjev")
    parser.add_argument("--task-mode", choices=TASK_MODES, default="direct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--full-param",
        action="store_true",
        help="Train ALL base-model parameters (full-parameter SFT) instead of a "
        "LoRA adapter. The checkpointer then writes a full standalone model dir "
        "instead of a PEFT adapter dir.",
    )
    parser.add_argument(
        "--lora-targets",
        default=",".join(DEFAULT_LORA_TARGETS),
        help="Comma-separated leaf module names.",
    )
    parser.add_argument("--low-ce-weight", type=float, default=0.5)
    parser.add_argument("--high-ce-weight", type=float, default=0.5)
    parser.add_argument("--distillation-weight", type=float, default=0.1)
    parser.add_argument("--laya-reward-weight", type=float, default=0.25)
    parser.add_argument("--hybrid-weight", type=float, default=0.25)
    parser.add_argument(
        "--fit-temperature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        help="Deterministic smoke-test limit; omit for complete training.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        help="Deterministic smoke-test limit; omit for complete evaluation.",
    )
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = local_rank = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OpenJev fine-tuning")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


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


def build_items(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    render_prompt: Any,
    label_mapping: Any,
    max_length: int,
    task_mode: str,
) -> list[dict[str, Any]]:
    return [
        compile_record(
            record,
            tokenizer,
            render_prompt=render_prompt,
            label_mapping=label_mapping,
            max_length=max_length,
        )
        for record in build_training_records(rows, task_mode)
    ]


def collate_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError(
            "the audited OpenJev early-exit executor requires --batch-size 1"
        )
    item = items[0]
    return {
        "input_ids": torch.tensor([item["input_ids"]], dtype=torch.long),
        "candidate_ids": torch.tensor(item["candidate_ids"], dtype=torch.long),
        "targets": torch.tensor([item["target"]], dtype=torch.long),
        "meta": item,
    }


class OpenJevTrainingModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        wrapper_type: type[torch.nn.Module],
        *,
        low_depth: int,
        high_depth: int,
        require_low_exit: bool,
    ) -> None:
        super().__init__()
        self.executor = wrapper_type(model)
        self.low_depth = low_depth
        self.high_depth = high_depth
        self.require_low_exit = require_low_exit

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        depths = (
            (self.low_depth, self.high_depth)
            if self.require_low_exit
            else (self.high_depth,)
        )
        decisions = self.executor(input_ids, candidate_ids, depths=depths)
        if self.require_low_exit:
            return decisions[0].logits, decisions[1].logits
        return None, decisions[0].logits


def make_loader(
    items: list[dict[str, Any]],
    *,
    shuffle: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = (
        DistributedSampler(
            items,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    return (
        DataLoader(
            items,
            batch_size=1,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            collate_fn=collate_items,
            num_workers=0,
            pin_memory=True,
        ),
        sampler,
    )


@torch.no_grad()
def evaluate(
    model: OpenJevTrainingModel,
    items: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model.eval()
    labels: list[int] = []
    low_probabilities: list[float] = []
    high_probabilities: list[float] = []
    high_logits_all = []
    targets_all = []
    for item in items:
        batch = collate_items([item])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            low_logits, high_logits = model(
                batch["input_ids"].to(device),
                batch["candidate_ids"].to(device),
            )
        target = int(batch["targets"].item())
        labels.append(target)
        high_probabilities.append(
            float(high_logits.float().softmax(-1)[0, 1].cpu())
        )
        if low_logits is not None:
            low_probabilities.append(
                float(low_logits.float().softmax(-1)[0, 1].cpu())
            )
        high_logits_all.append(high_logits.float().cpu())
        targets_all.append(batch["targets"])
    result = {"high": binary_metrics(labels, high_probabilities)}
    if low_probabilities:
        result["low"] = binary_metrics(labels, low_probabilities)
    return result, torch.cat(high_logits_all), torch.cat(targets_all)


def save_train_state(
    args: argparse.Namespace,
    target: Path,
    train_module: torch.nn.Module,
    tokenizer: Any,
    model_dir: Path,
) -> None:
    """Write checkpoints. LoRA -> PEFT adapter dir (merged later by
    merge_peft_adapter.py). Full-param -> complete standalone model dir carrying
    over the OpenJev runtime/prompt/template assets the eval harness needs."""
    target.mkdir(parents=True, exist_ok=True)
    if args.full_param:
        train_module.save_pretrained(str(target), safe_serialization=True)
        tokenizer.save_pretrained(target)
        _carryover_openjev_assets(model_dir, target)
    else:
        train_module.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)


def _carryover_openjev_assets(model_dir: Path, target: Path) -> None:
    """Copy OpenJev fast-path runtime + prompt/template assets so a full-param
    checkpoint remains loadable by evaluate_openjev_trait_probabilities.py."""
    import shutil

    for name in (
        "depth_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        src = model_dir / name
        if src.exists():
            dst = target / name
            if not dst.exists():
                shutil.copy2(src, dst)
    rt = model_dir / "openjet_runtime"
    if rt.is_dir():
        shutil.copytree(
            rt,
            target / "openjet_runtime",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _build_optimizer(
    args: argparse.Namespace,
    trainable: list[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    """AdamW with an fp32 master state. For --full-param the fp32 AdamW state on a
    ~9B model (~2 optimizer params * 4 bytes * ~8.7B params ~= 70 GiB) blows the
    single 79 GB A100 at the first optimizer.step(), so use 8-bit AdamW there
    (state fits in bf16/fp8 host buffers). LoRA keeps the standard fp32 AdamW."""
    if args.full_param:
        try:
            import bitsandbytes as bnb
        except Exception as err:  # pragma: no cover - env dependent
            raise RuntimeError(
                "full-param SFT needs bitsandbytes 8-bit AdamW (fp32 AdamW on a "
                f"~9B model OOMs the 79 GB GPU): {err}"
            ) from err
        return bnb.optim.AdamW8bit(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size must be 1 for the audited Q1 executor")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    rank, local_rank, world_size = distributed_context()
    seed_everything(args.seed, rank)
    device = torch.device("cuda", local_rank)
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    depth_config = json.loads(
        (model_dir / "depth_config.json").read_text(encoding="utf-8")
    )
    low_depth = int(depth_config["exit_depth"])
    high_depth = int(depth_config["full_depth"])
    render_prompt, label_mapping, wrapper_type = load_runtime(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    train_rows = load_rows(args.data_dir / "train.jsonl", args.max_train_rows)
    eval_rows = load_rows(args.data_dir / "eval.jsonl", args.max_eval_rows)
    train_items = build_items(
        train_rows,
        tokenizer,
        render_prompt,
        label_mapping,
        args.max_length,
        args.task_mode,
    )
    eval_items = build_items(
        eval_rows,
        tokenizer,
        render_prompt,
        label_mapping,
        args.max_length,
        "direct",
    )
    if not train_items or not eval_items:
        raise ValueError("train and eval splits must both be nonempty")

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise ValueError("this trainer currently requires Qwen3.5 conditional generation")
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    if args.full_param:
        train_module: torch.nn.Module = base_model
        for parameter in base_model.parameters():
            parameter.requires_grad = True
        # Gradient checkpointing drastically cuts activation memory so the 8-bit
        # AdamW optimizer state for a full ~9B fit alongside the model+grads on a
        # single 79 GB A100. (fp32 AdamW OOM'd at the first optimizer.step().)
        base_model.gradient_checkpointing_enable()
        lora_targets: list[str] = []
    else:
        lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        if not lora_targets:
            raise ValueError("--lora-targets must contain at least one module name")
        train_module = get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=lora_targets,
                bias="none",
            ),
        ).to(device)
    loss_config = LossConfig(
        mode=args.loss_mode,
        low_ce_weight=args.low_ce_weight,
        high_ce_weight=args.high_ce_weight,
        distillation_weight=args.distillation_weight,
        laya_reward_weight=args.laya_reward_weight,
        hybrid_weight=args.hybrid_weight,
    )
    model = OpenJevTrainingModel(
        train_module,
        wrapper_type,
        low_depth=low_depth,
        high_depth=high_depth,
        require_low_exit=args.loss_mode != "laya-reward",
    ).to(device)
    train_model: torch.nn.Module = (
        DDP(model, device_ids=[local_rank]) if world_size > 1 else model
    )
    trainable = [parameter for parameter in train_model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(
            "no trainable parameters; expected LoRA injection or --full-param"
        )
    optimizer = _build_optimizer(args, trainable)
    train_loader, train_sampler = make_loader(
        train_items,
        shuffle=True,
        rank=rank,
        world_size=world_size,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_optimizer_steps * args.warmup_ratio),
        num_training_steps=total_optimizer_steps,
    )

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_dir = output_dir / "best"
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                low_logits, high_logits = train_model(
                    batch["input_ids"].to(device),
                    batch["candidate_ids"].to(device),
                )
                loss, components = compute_training_loss(
                    low_logits,
                    high_logits,
                    batch["targets"].to(device),
                    loss_config,
                )
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            update = (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if update:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            eval_metrics, _, _ = evaluate(model, eval_items, device)
            epoch_result = {
                "epoch": epoch + 1,
                "train": {
                    name: value / len(train_loader)
                    for name, value in totals.items()
                },
                "eval": eval_metrics,
            }
            history.append(epoch_result)
            print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
            selection_f1 = eval_metrics["high"]["f1"]
            if selection_f1 > best_f1:
                best_f1 = selection_f1
                best_epoch = epoch + 1
                save_train_state(args, best_dir, train_module, tokenizer, model_dir)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        if args.full_param:
            eval_metrics, high_logits, eval_targets = evaluate(model, eval_items, device)
        else:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            adapter_state = load_file(best_dir / "adapter_model.safetensors")
            set_peft_model_state_dict(train_module, adapter_state)
            eval_metrics, high_logits, eval_targets = evaluate(model, eval_items, device)
        temperature = (
            fit_temperature(high_logits, eval_targets)
            if args.fit_temperature
            else 1.0
        )
        save_train_state(args, output_dir, train_module, tokenizer, model_dir)
        report = {
            "approximation_notice": (
                "Task-specific reproduction from the merged release; the private "
                "original curriculum, trainer state, and exact LoRA targets are unavailable."
            ),
            "base_model": str(model_dir),
            "train_rows": len(train_items),
            "eval_rows": len(eval_items),
            "train_source_rows": len(train_rows),
            "eval_source_rows": len(eval_rows),
            "task_mode": args.task_mode,
            "world_size": world_size,
            "loss": vars(loss_config),
            "method": "full-param" if args.full_param else "lora",
            "lora": None if args.full_param else {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "targets": lora_targets,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size_per_gpu": 1,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "seed": args.seed,
                "max_length": args.max_length,
                "low_depth": low_depth,
                "high_depth": high_depth,
            },
            "best_epoch": best_epoch,
            "temperature": temperature,
            "eval": eval_metrics,
            "history": history,
        }
        (output_dir / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
