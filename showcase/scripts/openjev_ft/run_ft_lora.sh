#!/bin/bash
# Args: $1 = mounted path of msn_dni_c09:/shares/users/wuc
# Fine-tune the merged OpenJev clickbait init with LoRA (quality_jev trainer),
# on the golden_set Track-B fallback split. Then merge the adapter back into the
# base and eval with the SAME harness as the merge sweep (golden_set.tsv +
# evaluate_openjev_trait_probabilities.py + sweep_thresholds.py) for a
# head-to-head plain-F1 comparison vs the block-e120-l000-c16 champion (0.5495).
set -euo pipefail
SHARE="$1"
CODE="$(pwd)"
OUT="${HOME}/ft-out"
FTDIR="$CODE/openjev_clickbait_ft"
mkdir -p "$OUT"

# ---- stage training code + data assets from the share ----
SRC_SHARE="$SHARE"
# Stage init to LOCAL disk once (the merge step's SECOND read of the RO share
# mount is flaky -> "OSError [Errno 5] Input/output error"). Train AND merge
# both point at the local copy so the ~14-shard base is only ever read locally.
INIT_MODEL="$OUT/init_local"
V12="$SRC_SHARE/openjev_clickbait/v12_training_set.tsv"
EVALDIR="$SRC_SHARE/openjev_clickbait"
echo "==== staging init model to local disk ===="
cp -r "$SRC_SHARE/qwen-clickbait-openjev-task-vector-50-0" "$OUT/init_local"
ls -la "$OUT/init_local" | head -20

# v12_training_set.tsv is ALREADY v12-schema (no golden conversion needed).
# Deterministic leakage-safe FULL multi-lingual train/eval split.
python "$FTDIR/prepare_laya_clickbait_finetune.py" \
  --input "$V12" \
  --output-dir "$OUT/data_all" \
  --language all \
  --seed 20260920


# ---- stage correct transformers + peft for OpenJev/Qwen3.5 load ----
python - "$OUT" << 'PYINST'
import subprocess, sys
out=sys.argv[1]
pkgs="transformers==5.16.1 peft==0.21.0 tokenizers==0.23.2"
# --no-deps: reuse the base env's consistent torch/torchvision instead of pulling
# a fresh torch into tflibs (torchvision mismatched -> torchvision::nms crash on
# modeling_utils import, masked as a bogus "Could not import module PreTrainedModel").
# tokenizers is pinned explicitly because --no-deps otherwise leaves the base env's
# 0.22.2, but transformers 5.16.1 requires >=0.23.1,<0.24.0.
r=subprocess.run([sys.executable,"-m","pip","install","--target",out+"/tflibs",
                  "--no-cache-dir","--no-deps"]+pkgs.split(),capture_output=True,text=True)
print(r.stdout[-1500:]); print("STDERR:",r.stderr[-1500:])
PYINST
export PYTHONPATH="$OUT/tflibs:$FTDIR:${PYTHONPATH:-}"

echo "==== LoRA finetune on merged init ===="
torchrun --standalone --nproc-per-node=1 \
  "$FTDIR/finetune_openjev_clickbait.py" \
  --data-dir "$OUT/data_all" \
  --model-dir "$INIT_MODEL" \
  --output-dir "$OUT/adapter_lora" \
  --loss-mode openjev \
  --task-mode direct-one-dimension \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --weight-decay 0 \
  --warmup-ratio 0 \
  --max-length 6144 \
  --seed 20260920 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0 \
  --low-ce-weight 0.5 \
  --high-ce-weight 0.5 \
  --distillation-weight 0.1 \
  --fit-temperature

echo "==== merge adapter back into init ===="
python "$FTDIR/merge_peft_adapter.py" \
  --base "$INIT_MODEL" \
  --adapter "$OUT/adapter_lora" \
  --output "$OUT/ft-lora-merged" \
  --device cuda:0

echo "==== EVAL ft-lora-merged on REAL v12 holdout ===="
(cd "$EVALDIR" && python -u "$EVALDIR/evaluate_openjev_trait_probabilities.py" \
  --model "$OUT/ft-lora-merged" --model-name "$(basename "$OUT/ft-lora-merged")-full" \
  --runtime-model "$OUT/ft-lora-merged" \
  --dataset "$SRC_SHARE/openjev_clickbait/golden_set.tsv" \
  --decision-prompt "$EVALDIR/openjev_clickbait_binary_prompt.txt" \
  --output-dir "$OUT/eval-lora" \
  --device cuda:0 --dtype bfloat16 --efforts high)
echo "==== plain threshold sweep ===="
python -u "$CODE/sweep_thresholds.py" "$OUT/eval-lora/ft-lora-merged-full-high.jsonl" \
  --metric plain --min 0.30 --max 0.70 --steps 801
echo "FINE-TUNE-LORA COMPLETE"

# ---- persist merged model + evals to the Gen2 share so they survive the node ----
PERSIST="${AZURE_ML_OUTPUT_MODEL:-}"
if [ -n "$PERSIST" ]; then
  echo "==== persisting LoRA-merged model + evals to Gen2: $PERSIST ===="
  mkdir -p "$PERSIST/model" "$PERSIST/eval"
  cp -r "$OUT/ft-lora-merged" "$PERSIST/model/"
  cp -r "$OUT/eval-lora" "$PERSIST/eval/"
  cp -r "$OUT/init_local" "$PERSIST/model/init_local" 2>/dev/null || true
  echo "==== persisted. contents: ===="
  ls -la "$PERSIST"
  du -sh "$PERSIST/model" "$PERSIST/eval" 2>/dev/null || true
else
  echo "WARN: AZURE_ML_OUTPUT_MODEL not set; LoRA model NOT persisted (stays in $OUT)"
fi
