#!/bin/bash
# Args: $1 = mounted path of msn_dni_c09:/shares/users/wuc
# PUSH v2 of full-param SFT on real v12. Champion v1 = 0.7964 (thr .5625, P .712 / R .903).
# Signature is recall-heavy -> precision drags F1. This run:
#   * 2 epochs (fit more)
#   * --pos-weight 0.7 (downweight positive class so the model is less trigger-happy
#     -> fewer false positives -> lifts precision, aims to raise F1)
# Writes to ft-sft-full-v2 (separate from champion v1). Same data/eval harness.
set -euo pipefail
SHARE="$1"
CODE="$(pwd)"
OUT="${HOME}/ft-sft-full-v2"
FTDIR="$CODE/openjev_clickbait_ft"
mkdir -p "$OUT"

SRC_SHARE="$SHARE"
INIT_MODEL="$OUT/init_local"
V12="$SRC_SHARE/openjev_clickbait/v12_training_set.tsv"
EVALDIR="$SRC_SHARE/openjev_clickbait"

echo "==== staging init model to local disk ===="
cp -r "$SRC_SHARE/qwen-clickbait-openjev-task-vector-50-0" "$OUT/init_local"
ls -la "$OUT/init_local" | head -20

# Real v12 is already v12-schema. Full multi-lingual leak-safe split.
python "$FTDIR/prepare_laya_clickbait_finetune.py" --input "$V12" \
  --output-dir "$OUT/data_all" --language all --seed 20260920


python - "$OUT" << 'PYINST'
import subprocess, sys
out=sys.argv[1]
pkgs="transformers==5.16.1 peft==0.21.0 tokenizers==0.23.2"
r=subprocess.run([sys.executable,"-m","pip","install","--target",out+"/tflibs",
                  "--no-cache-dir","--no-deps"]+pkgs.split(),capture_output=True,text=True)
print(r.stdout[-1500:]); print("STDERR:",r.stderr[-1500:])
PYINST
export PYTHONPATH="$OUT/tflibs:$FTDIR:${PYTHONPATH:-}"

echo "==== full-param SFT v2 (2 epochs, pos-weight 0.7) ===="
torchrun --standalone --nproc-per-node=1 \
  "$FTDIR/finetune_openjev_clickbait.py" \
  --data-dir "$OUT/data_all" \
  --model-dir "$INIT_MODEL" \
  --output-dir "$OUT/ft-sft-full-v2" \
  --loss-mode openjev \
  --task-mode direct-one-dimension \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-length 6144 \
  --seed 20260920 \
  --full-param \
  --low-ce-weight 0.5 \
  --high-ce-weight 0.9 \
  --distillation-weight 0.05 \
  --pos-weight 0.7 \
  --fit-temperature

echo "==== EVAL ft-sft-full-v2 on REAL v12 holdout ===="
(cd "$EVALDIR" && python -u "$EVALDIR/evaluate_openjev_trait_probabilities.py" \
  --model "$OUT/ft-sft-full-v2" --model-name "$(basename "$OUT/ft-sft-full-v2")-full" \
  --runtime-model "$OUT/ft-sft-full-v2" \
  --dataset "$SRC_SHARE/openjev_clickbait/golden_set.tsv" \
  --decision-prompt "$EVALDIR/openjev_clickbait_binary_prompt.txt" \
  --output-dir "$OUT/eval-sft-full-v2" --device cuda:0 --dtype bfloat16 --efforts high)
echo "==== plain threshold sweep ===="
python -u "$CODE/sweep_thresholds.py" "$OUT/eval-sft-full-v2/ft-sft-full-v2-full-high.jsonl" \
  --metric plain --min 0.30 --max 0.70 --steps 801
echo "FINE-TUNE-SFT-FULL-V2 COMPLETE"

# ---- persist model + evals to the Gen2 share so they survive the node ----
# $AZURE_ML_OUTPUT_MODEL is the RW_MOUNT output (msn_dni_c09:shares/users/wuc/openjev-ft/sft-full-v2)
PERSIST="${AZURE_ML_OUTPUT_MODEL:-$OUT/persist_disabled}"
if [ -n "${AZURE_ML_OUTPUT_MODEL:-}" ]; then
  echo "==== persisting model + evals to Gen2: $PERSIST ===="
  mkdir -p "$PERSIST/model" "$PERSIST/eval"
  # trained standalone SFT model (fully merged, no adapter needed)
  cp -r "$OUT/ft-sft-full-v2" "$PERSIST/model/"
  # eval jsonl + threshold sweep summary
  cp -r "$OUT/eval-sft-full-v2" "$PERSIST/eval/"
  cp -r "$OUT/init_local" "$PERSIST/model/init_local" 2>/dev/null || true
  echo "==== persisted. contents: ===="
  ls -la "$PERSIST"
  du -sh "$PERSIST/model" "$PERSIST/eval" 2>/dev/null || true
else
  echo "WARN: AZURE_ML_OUTPUT_MODEL not set; model NOT persisted (stays in $OUT)"
fi
