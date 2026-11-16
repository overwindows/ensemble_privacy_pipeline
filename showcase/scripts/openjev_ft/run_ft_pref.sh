#!/bin/bash
# Args: $1 = mounted path of msn_dni_c09:/shares/users/wuc; $2 = method (dpo|rlcd)
# Fine-tune the merged OpenJev clickbait init with DPO or RLCD (preference_trainer.py)
# on the golden_set Track-B split (same data-prep + eval harness as the LoRA run).
# policy = init; reference = untouched init (frozen). Merge adapter + sweep thresholds.
set -euo pipefail
SHARE="$1"
METHOD="$2"
CODE="$(pwd)"
OUT="${HOME}/ft-pref-${METHOD}"
FTDIR="$CODE/openjev_clickbait_ft"
mkdir -p "$OUT"

SRC_SHARE="$SHARE"
# Stage init to LOCAL disk once (merge step's SECOND read of the RO share mount
# is flaky -> "OSError [Errno 5] Input/output error"). Train AND merge both
# point at the local copy so the ~14-shard base is only read locally.
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
# --no-deps: reuse base env torch/torchvision (staging a fresh torch into tflibs
# mismatches torchvision -> torchvision::nms crash, masked as bogus PreTrainedModel error).
# tokenizers pinned: transformers 5.16.1 needs >=0.23.1,<0.24.0 (base env has 0.22.2).
r=subprocess.run([sys.executable,"-m","pip","install","--target",out+"/tflibs",
                  "--no-cache-dir","--no-deps"]+pkgs.split(),capture_output=True,text=True)
print(r.stdout[-1500:]); print("STDERR:",r.stderr[-1500:])
PYINST
export PYTHONPATH="$OUT/tflibs:$FTDIR:${PYTHONPATH:-}"

echo "==== ${METHOD} finetune on merged init (ref=frozen init) ===="
torchrun --standalone --nproc-per-node=1 \
  "$FTDIR/preference_trainer.py" \
  --data-dir "$OUT/data_all" \
  --model-dir "$INIT_MODEL" \
  --ref-model-dir "$INIT_MODEL" \
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

echo "==== merge adapter back into init ===="
python "$FTDIR/merge_peft_adapter.py" --base "$INIT_MODEL" \
  --adapter "$OUT/adapter_${METHOD}" --output "$OUT/ft-${METHOD}-merged" --device cuda:0

echo "==== EVAL ft-${METHOD}-merged on REAL v12 holdout ===="
(cd "$EVALDIR" && python -u "$EVALDIR/evaluate_openjev_trait_probabilities.py" \
  --model "$OUT/ft-${METHOD}-merged" --model-name "$(basename "$OUT/ft-${METHOD}-merged")-full" \
  --runtime-model "$OUT/ft-${METHOD}-merged" \
  --dataset "$SRC_SHARE/openjev_clickbait/golden_set.tsv" \
  --decision-prompt "$EVALDIR/openjev_clickbait_binary_prompt.txt" \
  --output-dir "$OUT/eval-${METHOD}" --device cuda:0 --dtype bfloat16 --efforts high)
echo "==== plain threshold sweep ===="
python -u "$CODE/sweep_thresholds.py" "$OUT/eval-${METHOD}/ft-${METHOD}-merged-full-high.jsonl" \
  --metric plain --min 0.30 --max 0.70 --steps 801
echo "FINE-TUNE-${METHOD} COMPLETE"

# ---- persist merged model + evals to the Gen2 share so they survive the node ----
PERSIST="${AZURE_ML_OUTPUT_MODEL:-}"
if [ -n "$PERSIST" ]; then
  echo "==== persisting ${METHOD}-merged model + evals to Gen2: $PERSIST ===="
  mkdir -p "$PERSIST/model" "$PERSIST/eval"
  cp -r "$OUT/ft-${METHOD}-merged" "$PERSIST/model/"
  cp -r "$OUT/eval-${METHOD}" "$PERSIST/eval/"
  cp -r "$OUT/init_local" "$PERSIST/model/init_local" 2>/dev/null || true
  echo "==== persisted. contents: ===="
  ls -la "$PERSIST"
  du -sh "$PERSIST/model" "$PERSIST/eval" 2>/dev/null || true
else
  echo "WARN: AZURE_ML_OUTPUT_MODEL not set; ${METHOD} model NOT persisted (stays in $OUT)"
fi
