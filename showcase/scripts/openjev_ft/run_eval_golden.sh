#!/bin/bash
# Args: $1 = mounted path of msn_dni_c09:/shares/users/wuc
# Score the 6 persisted fine-tuned OpenJev models on the FULL golden_set.tsv
# (11,914 rows, could not be more different from the v12 eval path: golden_set is
# Minglei/Xinyi's official labeled quality set used to benchmark the PROD domain
# model qwen_clickbait_production_20260629_2 => plain-F1 0.5818 / weighted-F1 0.6947).
# This produces an apples-to-apples OpenJev-finetune vs prod-model head-to-head on
# the SAME dataset the prod model was measured on. Eval-only: no training.
# Eval harness = evaluate_openjev_trait_probabilities.py + sweep_thresholds.py,
# identical to the merge sweep so numbers are directly comparable to 0.5495.
set -euo pipefail
SHARE="$1"
CODE="$(pwd)"
OUT="${HOME}/golden-eval"
FTDIR="$CODE/openjev_clickbait_ft"
SRC_SHARE="$SHARE/openjev_clickbait"
mkdir -p "$OUT"

GOLDEN="$SRC_SHARE/golden_set.tsv"
PROMPT="$SRC_SHARE/openjev_clickbait_binary_prompt.txt"

# The 6 persisted merged models + their share-relative subdirs.
# $SHARE = shares/users/wuc. Models persist to
# shares/users/wuc/openjev-ft/<method>/model/<merged> (training launchers write
# $AZURE_ML_OUTPUT_MODEL=.../openjev-ft/<method>, run_ft_*.sh copies to model/).
declare -A MODEL_DIR=(
  [lora]="openjev-ft/lora/model/ft-lora-merged"
  [sft-v2]="openjev-ft/sft-full-v2/model/ft-sft-full-v2"
  [dpo]="openjev-ft/dpo/model/ft-dpo-merged"
  [rlcd]="openjev-ft/rlcd/model/ft-rlcd-merged"
  [dpo-on-lora]="openjev-ft/dpo-on-lora/model/ft-dpo-lora-merged"
  [rlcd-on-lora]="openjev-ft/rlcd-on-lora/model/ft-rlcd-lora-merged"
)

echo "==== staging golden_set.tsv + prompt ===="
ls -la "$SRC_SHARE" | head

# Pre-flight: verify every persisted model exists on the share BEFORE staging so
# we fail fast on a path typo instead of mid-copy (the earlier v1 failed here).
for name in "${!MODEL_DIR[@]}"; do
  if [ ! -e "$SHARE/${MODEL_DIR[$name]}" ]; then
    echo "ERROR: persisted model not found: $SHARE/${MODEL_DIR[$name]}" >&2
    exit 1
  fi
done

# Stage each merged model to LOCAL disk ONCE (RO share mount second reads are
# flaky -> "OSError [Errno 5]"); eval reads the local copy.
mkdir -p "$OUT/model"
for name in "${!MODEL_DIR[@]}"; do
  rel="${MODEL_DIR[$name]}"
  dst="$OUT/model/$name"
  echo "==== stage $name <- $rel ===="
  cp -r "$SHARE/$rel" "$dst"
  ls -la "$dst" | head -8
done

# correct transformers + peft for OpenJev/Qwen3.5 load (same as training jobs)
python - "$OUT" << 'PYINST'
import subprocess, sys
out=sys.argv[1]
pkgs="transformers==5.16.1 peft==0.21.0 tokenizers==0.23.2"
r=subprocess.run([sys.executable,"-m","pip","install","--target",out+"/tflibs",
                  "--no-cache-dir","--no-deps"]+pkgs.split(),capture_output=True,text=True)
print(r.stdout[-1500:]); print("STDERR:",r.stderr[-1500:])
PYINST
export PYTHONPATH="$OUT/tflibs:$FTDIR:${PYTHONPATH:-}"

for name in "${!MODEL_DIR[@]}"; do
  M="$OUT/model/$name"
  echo "=========== EVAL $name on golden_set (full 11,914) ==========="
  (cd "$SRC_SHARE" && python -u "$SRC_SHARE/evaluate_openjev_trait_probabilities.py" \
    --model "$M" --model-name "$name-golden" \
    --runtime-model "$M" \
    --dataset "$GOLDEN" \
    --decision-prompt "$PROMPT" \
    --output-dir "$OUT/eval-$name" --device cuda:0 --dtype bfloat16 --efforts high)
  echo "==== plain + weighted threshold sweep ($name) ===="
  python -u "$CODE/sweep_thresholds.py" "$OUT/eval-$name/$name-golden-high.jsonl" \
    --metric plain --min 0.30 --max 0.70 --steps 801
  python -u "$CODE/sweep_thresholds.py" "$OUT/eval-$name/$name-golden-high.jsonl" \
    --metric weighted --min 0.30 --max 0.70 --steps 801
done

echo "==== GOLDEN_SET EVAL ALL COMPLETE ===="
# coalesced summary table
python - "$OUT" << 'PYSUM'
import glob, json, os, sys
out=sys.argv[1]
print(f"{'method':<14}{'rows':<7}{'plainF1@thr':<22}{'weightedF1@thr':<24}")
for jl in sorted(glob.glob(f"{out}/eval-*/*-golden-high.jsonl")):
    name=os.path.basename(jl).replace("-golden-high.jsonl","")
    recs=[json.loads(l) for l in open(jl,encoding='utf-8') if l.strip()]
    print(f"  {name:<14} scored {len(recs)} rows -> thresholds below")
PYSUM
PY

echo "GOLDEN EVAL DONE"
