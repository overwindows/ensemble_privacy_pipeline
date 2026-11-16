#!/bin/bash
# Args: $1 = mounted path of msn_dni_c09:/shares/users/wuc ; $2 = method (dpo|rlcd)
# STACKED preference run: policy = the persisted full-param SFT model on the share,
# reference = untouched raw merged init (frozen). This is the "DPO/RLCD on the 0.79
# SFT" experiment Chen asked for — the SFT base must have been produced+persisted by
# the SFT-full-v2 job (AZURE_ML_OUTPUT_MODEL -> openjev-ft/sft-full-v2/model).
set -euo pipefail
SHARE="$1"
METHOD="$2"
CODE="$(pwd)"
OUT="${HOME}/ft-pref-${METHOD}-sft"
FTDIR="$CODE/openjev_clickbait_ft"
mkdir -p "$OUT"

SRC_SHARE="$SHARE"
# Policy = persisted SFT model from the share (produced by the SFT-full-v2 run).
# Reference = raw merged init (frozen, untouched). Stage both to LOCAL disk to
# avoid the flaky second-read of the RO share mount (OSError Errno 5).
SFT_BASE="$SRC_SHARE/openjev-ft/sft-full-v2/model/ft-sft-full-v2"
INIT_RAW="$SRC_SHARE/qwen-clickbait-openjev-task-vector-50-0"
V12="$SRC_SHARE/openjev_clickbait/v12_training_set.tsv"
EVALDIR="$SRC_SHARE/openjev_clickbait"

echo "==== staging policy (SFT) + reference (raw init) to local disk ===="
if [ ! -d "$SFT_BASE" ]; then
  echo "ERROR: persisted SFT policy not found at $SFT_BASE"
  echo "SFT-full-v2 must complete + persist before stacked ${METHOD} can run."
  ls -la "$SRC_SHARE/openjev-ft/" 2>/dev/null || true
  exit 1
fi
cp -r "$SFT_BASE" "$OUT/policy_sft"
cp -r "$INIT_RAW" "$OUT/ref_init"
INIT_MODEL="$OUT/policy_sft"
REF_MODEL="$OUT/ref_init"
ls -la "$INIT_MODEL" | head -20

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

echo "==== STACKED ${METHOD} finetune on SFT-policy (ref=frozen raw init) ===="
torchrun --standalone --nproc-per-node=1 \
  "$FTDIR/preference_trainer.py" \
  --data-dir "$OUT/data_all" \
  --model-dir "$INIT_MODEL" \
  --ref-model-dir "$REF_MODEL" \
  --output-dir "$OUT/adapter_${METHOD}" \
  --method "$METHOD" \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --weight-decay 0 \
  --warmup-ratio 0 \
  --max-length 6144 \
  --seed 20260920 \
  --beta 0.1 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0

echo "==== merge STACKED adapter back into SFT policy ===="
python "$FTDIR/merge_peft_adapter.py" --base "$INIT_MODEL" \
  --adapter "$OUT/adapter_${METHOD}" --output "$OUT/ft-${METHOD}-sft-merged" --device cuda:0

echo "==== EVAL stacked ft-${METHOD}-sft-merged on REAL v12 holdout ===="
(cd "$EVALDIR" && python -u "$EVALDIR/evaluate_openjev_trait_probabilities.py" \
  --model "$OUT/ft-${METHOD}-sft-merged" --model-name "$(basename "$OUT/ft-${METHOD}-sft-merged")-full" \
  --runtime-model "$OUT/ft-${METHOD}-sft-merged" \
  --dataset "$SRC_SHARE/openjev_clickbait/golden_set.tsv" \
  --decision-prompt "$EVALDIR/openjev_clickbait_binary_prompt.txt" \
  --output-dir "$OUT/eval-${METHOD}" --device cuda:0 --dtype bfloat16 --efforts high)
echo "==== plain threshold sweep ===="
python -u "$CODE/sweep_thresholds.py" "$OUT/eval-${METHOD}/ft-${METHOD}-sft-merged-full-high.jsonl" \
  --metric plain --min 0.30 --max 0.70 --steps 801
echo "FINE-TUNE-${METHOD}-ON-SFT COMPLETE"

# ---- persist merged model + evals to the Gen2 share so they survive the node ----
PERSIST="${AZURE_ML_OUTPUT_MODEL:-}"
if [ -n "$PERSIST" ]; then
  echo "==== persisting stacked ${METHOD}-on-SFT model + evals to Gen2: $PERSIST ===="
  mkdir -p "$PERSIST/model" "$PERSIST/eval"
  cp -r "$OUT/ft-${METHOD}-sft-merged" "$PERSIST/model/"
  cp -r "$OUT/eval-${METHOD}" "$PERSIST/eval/"
  cp -r "$OUT/policy_sft" "$PERSIST/model/policy_sft" 2>/dev/null || true
  echo "==== persisted. contents: ===="
  ls -la "$PERSIST"
  du -sh "$PERSIST/model" "$PERSIST/eval" 2>/dev/null || true
else
  echo "WARN: AZURE_ML_OUTPUT_MODEL not set; stacked ${METHOD} model NOT persisted (stays in $OUT)"
fi