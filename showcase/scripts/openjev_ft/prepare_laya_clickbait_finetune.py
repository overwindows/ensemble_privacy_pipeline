#!/usr/bin/env python3
"""Prepare deterministic English train/eval splits for Laya clickbait tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "ContentId",
    "Market",
    "ContentType",
    "Headline",
    "clickbait_score",
    "el_score",
    "et_score",
    "po_score",
    "pr_score",
    "category",
}
DIMENSIONS = ("el_score", "et_score", "po_score", "pr_score")


def normalize_headline(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def canonical_market(value: str) -> str:
    parts = value.strip().replace("_", "-").split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return value.strip()


def load_rows(path: Path, language: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        rows = list(reader)
    if language == "en":
        return [
            row
            for row in rows
            if canonical_market(row["Market"]) == "en-US"
        ]
    return rows


def aggregate_headlines(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        headline_key = normalize_headline(row["Headline"])
        if not headline_key:
            raise ValueError("headline is empty after normalization")
        market = canonical_market(row["Market"])
        key = (market, headline_key)
        grouped.setdefault(key, []).append(row)

    prepared = []
    for (market, headline_key), group in sorted(grouped.items()):
        scores = [float(row["clickbait_score"]) for row in group]
        score = float(statistics.median(scores))
        dimension_scores = {
            dimension: float(
                statistics.median(float(row[dimension]) for row in group)
            )
            for dimension in DIMENSIONS
        }
        first = group[0]
        prepared.append(
            {
                "id": hashlib.sha256(
                    f"{market}\n{headline_key}".encode("utf-8")
                ).hexdigest()[:16],
                "headline": first["Headline"].strip(),
                "market": market,
                "content_type": Counter(
                    row["ContentType"].strip() for row in group
                ).most_common(1)[0][0],
                "score": score,
                "label": int(score >= 50),
                "dimension_scores": dimension_scores,
                "dimension_labels": {
                    dimension: int(value >= 50)
                    for dimension, value in dimension_scores.items()
                },
                "source": Counter(
                    row["category"].strip() for row in group
                ).most_common(1)[0][0],
                "annotation_count": len(group),
            }
        )
    return prepared


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")
    rng = random.Random(seed)
    train, evaluation = [], []
    strata = sorted(
        {(row["market"], row["source"], row["label"]) for row in rows}
    )
    for market, source, label in strata:
        group = [
            row
            for row in rows
            if row["market"] == market
            and row["source"] == source
            and row["label"] == label
        ]
        rng.shuffle(group)
        eval_count = max(1, round(len(group) * eval_ratio))
        evaluation.extend(group[:eval_count])
        train.extend(group[eval_count:])
    rng.shuffle(train)
    rng.shuffle(evaluation)
    if {row["id"] for row in train} & {row["id"] for row in evaluation}:
        raise AssertionError("headline leakage detected across train/eval")
    return train, evaluation


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(row["label"] for row in rows)
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
        "positive_rate": positives / len(rows),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "markets": dict(sorted(Counter(row["market"] for row in rows).items())),
    }


def main() -> int:
    default_root = os.environ.get("AZURE_ML_INPUT_ukwdata")
    default_input = (
        Path(default_root) / "quality/data/v12_training_set.tsv"
        if default_root
        else None
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", choices=("en", "all"), default="en")
    args = parser.parse_args()
    if args.input is None:
        parser.error("--input is required when AZURE_ML_INPUT_ukwdata is unset")

    source_rows = load_rows(args.input, args.language)
    prepared = aggregate_headlines(source_rows)
    train, evaluation = split_rows(
        prepared,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(evaluation, args.output_dir / "eval.jsonl")
    manifest = {
        "source": str(args.input.resolve()),
        "source_rows": len(source_rows),
        "language": args.language,
        "market_values": sorted(
            {canonical_market(row["Market"]) for row in source_rows}
        ),
        "unique_headlines": len(prepared),
        "duplicates_collapsed": len(source_rows) - len(prepared),
        "label_rule": "clickbait_score >= 50",
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "splits": {
            "train": distribution(train),
            "eval": distribution(evaluation),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
