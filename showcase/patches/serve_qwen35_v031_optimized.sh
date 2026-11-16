#!/usr/bin/env bash
# ==============================================================================
# Qwen3.5-4B Online HTTP (/v1/chat/completions) Production Serving Runbook Launcher
# Runtime: vLLM 0.31.0 (V2 Model Runner) on 1x NVIDIA A100-80GB (sm_80)
#
# Verified Full-2000 Online HTTP /v1/chat/completions Benchmarks:
#   - Champion (Strict Greedy T=0.0, 4-Patch Stack): 3,879.50 output tok/s
#     (magenta_book_0g0hp7t2cy | +92.66 tok/s / +2.45% vs r9 | 12,728.44 total tok/s | 3.01 QPS | 663.54s | 2000/2000 JSON)
#   - 3-Patch Stack (Fastest Wall Time, Greedy T=0.0): 3,877.46 output tok/s
#     (frank_jelly_p13140lkhv | +90.62 tok/s / +2.39% vs r9 | 12,741.56 total tok/s | 3.02 QPS | 662.41s | 2000/2000 JSON)
#   - Stock vLLM 0.31.0 (Strict Greedy T=0.0, 0 Patches): 3,843.12 output tok/s
#     (hungry_cat_jqbhqqvz0z | +56.28 tok/s / +1.49% vs r9 | 12,606.40 total tok/s | 2.98 QPS | 670.03s | 1998/2000 JSON)
#   - Champion 4-Patch Stack (Stochastic T=0.2): 3,827.28 output tok/s
#     (ivory_box_09wc1wnqzl | +40.44 tok/s / +1.07% vs r9 | 12,519.51 total tok/s | 2.96 QPS | 675.51s | 1998/2000 JSON)
# ==============================================================================
set -euo pipefail

port="${1:-${VLLM_PORT:-8100}}"
data_parallel="${2:-${VLLM_DATA_PARALLEL_SIZE:-1}}"
model_path="${MODEL_PATH:-${3:-$PWD}}"
served_model_name="${SERVED_MODEL_NAME:-maiprofile-qwen35-sft}"
tensor_parallel="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
max_model_len="${VLLM_MAX_MODEL_LEN:-25600}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
max_num_seqs="${VLLM_MAX_NUM_SEQS:-128}"
max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
speculative_method="${VLLM_SPECULATIVE_METHOD:-mtp}"
speculative_tokens="${VLLM_NUM_SPECULATIVE_TOKENS:-3}"
draft_model_path="${VLLM_DRAFT_MODEL_PATH:-}"
draft_attention_backend="${VLLM_DRAFT_ATTENTION_BACKEND:-TRITON_ATTN}"
dspark_draft_topk="${VLLM_DSPARK_DRAFT_TOPK:-0}"
api_server_count="${VLLM_API_SERVER_COUNT:-1}"
enable_prefix_caching="${VLLM_ENABLE_PREFIX_CACHING:-false}"
mamba_cache_mode="${VLLM_MAMBA_CACHE_MODE:-align}"
enable_reasoning_parser="${VLLM_ENABLE_REASONING_PARSER:-true}"
patch_dir="${VLLM_PATCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
apply_patches="${VLLM_APPLY_PATCHES:-auto}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Optimal SM80 D2G Triton unified_attention parameters (effective when qwen35_d2g_v031.patch is applied)
export D2G_TILE="${D2G_TILE:-64}"
export D2G_STAGES="${D2G_STAGES:-2}"
export D2G_WARPS="${D2G_WARPS:-8}"

# Automatically apply the Wave-6g 3,879.50 tok/s 4-patch stack to the installed vLLM 0.31.0 site-packages if present
if [[ "$apply_patches" == "auto" || "$apply_patches" == "true" || "$apply_patches" == "1" ]]; then
    vllm_site_root="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))' 2>/dev/null || true)"
    if [[ -n "$vllm_site_root" && -d "$vllm_site_root/vllm" ]]; then
        champion_patches=(
            "vllm_v031_eagle_full_cudagraph.patch"
            "vllm_v031_fast_spec_sample.patch"
            "vllm_v031_gdn_minimal.patch"
            "qwen35_d2g_v031.patch"
        )
        if [[ "$speculative_method" == "dspark" ]]; then
            champion_patches+=("vllm_v031_dspark_kv_groups_fix.patch" "vllm_v031_dspark_fast_topk_greedy.patch")
        fi
        for p in "${champion_patches[@]}"; do
            pf="$patch_dir/$p"
            if [[ -f "$pf" ]]; then
                if patch -d "$vllm_site_root" -p1 --forward --dry-run < "$pf" >/dev/null 2>&1; then
                    patch -d "$vllm_site_root" -p1 --forward < "$pf" >/dev/null
                    echo "[wave6g-patch] Applied $p to $vllm_site_root"
                else
                    echo "[wave6g-patch] $p already applied or skipped in $vllm_site_root"
                fi
            fi
        done
    fi
fi

# If using DSpark external draft model, automatically patch its config.json to avoid
# the vLLM 0.31.0 34-KV-cache-group fragmentation bug (collapses 34 groups -> 4 groups).
if [[ "$speculative_method" == "dspark" && -n "$draft_model_path" && -f "$draft_model_path/config.json" ]]; then
    python3 - "$draft_model_path/config.json" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p, "r", encoding="utf-8"))
if cfg.get("layer_types") == ["sliding_attention", "full_attention"]:
    cfg["layer_types"] = ["full_attention", "full_attention"]
    cfg["sliding_window"] = None
    cfg["use_sliding_window"] = False
    json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[dspark-4group-fix] Patched {p}: layer_types -> full_attention (collapses 34 KV groups -> 4)")
PY
fi

speculative_args=()
if (( speculative_tokens > 0 )); then
    if [[ "$speculative_method" == "dspark" && -n "$draft_model_path" ]]; then
        if (( dspark_draft_topk > 0 )); then
            speculative_args=(--speculative-config "{\"method\": \"dspark\", \"model\": \"${draft_model_path}\", \"num_speculative_tokens\": ${speculative_tokens}, \"attention_backend\": \"${draft_attention_backend}\", \"dspark_draft_topk\": ${dspark_draft_topk}}")
        else
            speculative_args=(--speculative-config "{\"method\": \"dspark\", \"model\": \"${draft_model_path}\", \"num_speculative_tokens\": ${speculative_tokens}, \"attention_backend\": \"${draft_attention_backend}\"}")
        fi
    else
        speculative_args=(--speculative-config "{\"method\": \"${speculative_method}\", \"num_speculative_tokens\": ${speculative_tokens}, \"attention_backend\": \"${draft_attention_backend}\"}")
    fi
fi

prefix_cache_args=(--no-enable-prefix-caching)
if [[ "${enable_prefix_caching,,}" == "true" || "$enable_prefix_caching" == "1" ]]; then
    prefix_cache_args=(--enable-prefix-caching)
    if [[ -n "$mamba_cache_mode" ]]; then
        prefix_cache_args+=(--mamba-cache-mode "$mamba_cache_mode")
    fi
fi

reasoning_args=()
if [[ "${enable_reasoning_parser,,}" == "true" || "$enable_reasoning_parser" == "1" ]]; then
    reasoning_args=(--reasoning-parser qwen3)
fi

api_server_args=()
if (( api_server_count > 1 )); then
    api_server_args=(--api-server-count "$api_server_count")
fi

exec vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port "$port" \
    --served-model-name "$served_model_name" \
    --tensor-parallel-size "$tensor_parallel" \
    --data-parallel-size "$data_parallel" \
    --max-model-len "$max_model_len" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --dtype bfloat16 \
    --trust-remote-code \
    --hf-overrides '{"architectures": ["Qwen3_5ForConditionalGeneration"]}' \
    --async-scheduling \
    --no-enable-log-requests \
    --disable-uvicorn-access-log \
    --mamba-ssm-cache-dtype bfloat16 \
    "${prefix_cache_args[@]}" \
    --attention-backend FLASHINFER \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --language-model-only \
    ${reasoning_args[@]+"${reasoning_args[@]}"} \
    ${api_server_args[@]+"${api_server_args[@]}"} \
    ${speculative_args[@]+"${speculative_args[@]}"}
