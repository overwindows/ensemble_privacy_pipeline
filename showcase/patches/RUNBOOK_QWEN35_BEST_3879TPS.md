# Production Serving & Reproduction Runbook — Qwen3.5-4B (`maiprofile-qwen35-sft`) at `3,879.50 output tok/s`

- **Model:** `maiprofile-qwen35-sft` (`ChenWu/qwen35-l1234-v25-step6000` / `MAI-Profile-Qwen35-SFT-step12000`, Qwen3.5-4B hybrid 24×GDN + 8×Full-Attention + 1-layer native MTP)
- **Hardware & Runtime:** 1× NVIDIA A100-SXM4-80GB (`sm_80`), `vLLM 0.31.0` V2 Model Runner (`pip install vllm==0.31.0`, `torch 2.13.0+cu130`, FlashInfer + Triton, AML environment `azureml:vllm_gemma4:9`)
- **Serving Surface:** Online HTTP `/v1/chat/completions` (`vllm serve`)
- **Verified Full-2000 Production Benchmark (`l1_v2_gpt56_test.jsonl`, 2,000 requests, `5,871,663` prompt tokens, `2,574,218` output tokens):**
  - **Champion (`HTTP_V031_WAVE6G_GREEDY_MINIMAL_STACK_K3`, `magenta_book_0g0hp7t2cy`):** **`3,879.50 output tok/s`** (**`+92.66 tok/s` / `+2.45%`** over `r9` `3,786.84 tok/s`), **`12,728.44 total tok/s`**, **`3.01 QPS`**, **`663.54s` wall time** (`-18.36s` vs `r9`), **`2000/2000 (100.00%)` JSON valid**, **`0.7664`** MTP draft acceptance (`0.885 / 0.758 / 0.648`, mean accept length **`3.30`** tokens/step).
  - **3-Patch Stack (`HTTP_V031_WAVE6G_GREEDY_EAGLE_FULLCG_SAMPLE_K3`, `frank_jelly_p13140lkhv`):** **`3,877.46 output tok/s`**, **`12,741.56 total tok/s`**, **`3.02 QPS`**, **`662.41s` wall time** (`-19.49s` vs `r9`), **`2000/2000 (100.00%)` JSON valid**.
  - **Zero-Patch Stock Control (`HTTP_V031_WAVE6F_GREEDY_R9_CTRL`, `hungry_cat_jqbhqqvz0z`):** **`3,843.12 output tok/s`** (**`+56.28 tok/s` / `+1.49%`** over `r9`), **`12,606.40 total tok/s`**, **`2.98 QPS`**, **`670.03s` wall time**, **`1998/2000 (99.90%)` JSON valid**.
  - **Champion 4-Patch Stack (`T=0.2` Stochastic, `ivory_box_09wc1wnqzl`):** **`3,827.28 output tok/s`** (**`+40.44 tok/s` / `+1.07%`** over `r9`), **`12,519.51 total tok/s`**, **`2.96 QPS`**, **`675.51s` wall time**, **`1998/2000 (99.90%)` JSON valid**.

---

## 1. Complete Best Settings Matrix

| Layer | Parameter / Flag | Champion Value (`3,879.50 tok/s`) | Why It Matters (Measured Impact) |
|---|---|---|---|
| **Speculative Decoding** | `--speculative-config` | `'{"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "TRITON_ATTN"}'` | Native 1-layer MTP (`k=3`) beats `k=4` by **+132.5 tok/s** and external 2L DSpark (`3,703.97 tok/s`) by **+175.5 tok/s** due to 1-layer draft weight footprint + shared target activations. |
| **Target Attention** | `--attention-backend` | `FLASHINFER` | Beats `TRITON_ATTN` target by **+129 tok/s** (`+3.45%`) and `FLASH_ATTN` target by **+1,311 tok/s** (`+51%` on `head_dim=256`). |
| **Mamba / GDN State** | `--mamba-ssm-cache-dtype` | `bfloat16` | Cuts recurrent GDN state memory bandwidth 2× vs FP32. |
| **Prefix Caching** | `--no-enable-prefix-caching` | Disabled (`false`) | Enabling `--mamba-cache-mode align` inflates attention KV block size `16 -> 288` tokens and costs **-214.8 tok/s (-5.66%)** on low-prefix-hit workloads. |
| **Batch Scheduling** | `--max-num-seqs` / `--max-num-batched-tokens` | `128` / `16384` | Sweet spot for 1×A100-80GB: avoids the 32k-token FLA `chunk_gated_delta_rule_fwd` workspace OOM while beating `mnbt=8192` by **+57.1 tok/s** and `mns=256` by **+58.4 tok/s**. |
| **Context & Memory** | `--max-model-len` / `--gpu-memory-utilization` | `25600` / `0.9` | Leaves ~8 GiB headroom for FLA chunked-prefill activations and CUDA graph pools. |
| **Multimodal & Thinking** | `--language-model-only` + `--default-chat-template-kwargs` | `'{"enable_thinking": false}'` + `--reasoning-parser qwen3` | Skips vision encoder weights and prevents `<think>` chain-of-thought token inflation (`1,287` vs `2,252` tok/req, **+23.7% TPS**). |
| **Server Overhead** | `--async-scheduling --no-enable-log-requests --disable-uvicorn-access-log` | Enabled (`api_server_count=1`) | Eliminates Python HTTP/logger IPC bottlenecks; 1 API worker beats 4 workers by **+56.2 tok/s**. |
| **Request Sampling** | `temperature` / `top_p` | `temperature: 0.0, top_p: 1.0` | **+49.9 to +90.3 tok/s (+1.32% to +2.47%)** over `T=0.2`; lifts MTP draft acceptance (`0.7631 -> 0.7664+`) and makes BF16 zero-copy rejection sampling **100% mathematically bitwise-exact**. |
| **Environment Vars** | `D2G_TILE` / `D2G_STAGES` / `D2G_WARPS` | `D2G_TILE=64 D2G_STAGES=2 D2G_WARPS=8` | Best SM80 Triton `unified_attention` geometry for the `TRITON_ATTN` draft head (+41.2 tok/s over `TILE=32, STAGES=2`). |

---

## 2. Step-by-Step Deployment Procedure

### Step 1 — Apply the 4 Pure-Python Patches to `vLLM 0.31.0` (No C++/CUDA Rebuild Required)
All four patches in the `3,879.50 tok/s` champion stack touch only `.py` files and Triton `@triton.jit` kernels inside `site-packages/vllm` — they apply in `<1 second` at container start with **zero C++/CUDA compilation**:

```bash
VLLM_SITE_ROOT="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__))))')"

patch -d "$VLLM_SITE_ROOT" -p1 --forward < vllm_v031_eagle_full_cudagraph.patch
patch -d "$VLLM_SITE_ROOT" -p1 --forward < vllm_v031_fast_spec_sample.patch
patch -d "$VLLM_SITE_ROOT" -p1 --forward < vllm_v031_gdn_minimal.patch
patch -d "$VLLM_SITE_ROOT" -p1 --forward < qwen35_d2g_v031.patch
```

What each patch does:
1. **`vllm_v031_eagle_full_cudagraph.patch` (`+23.13 tok/s` greedy lift):** Unlocks `CUDAGraphMode.FULL_DECODE_ONLY` on `AutoRegressiveSpeculator` (capturing fused multi-step MTP draft decode `_generate_fused_drafts` into a single CUDA graph per batch size) with `TRITON_ATTN` draft attention alongside `FLASHINFER` (`PIECEWISE`) target attention.
2. **`vllm_v031_fast_spec_sample.patch` (`+31.04 tok/s` lift, 100% bitwise-exact under greedy):** Bypasses the unconditional `[512, 248320]` `BF16 -> FP32` logits copy (`508.5 MB` alloc + `762 MB` copy) and standalone `_temperature_kernel` (`1.02 GB` R/W) when no non-temperature logit processors (`top_k`, `top_p < 1.0`, `min_p`, penalties, bad_words) are active, cutting rejection-sampling HBM traffic by **88% (`2.03 GB` saved per step)**.
3. **`vllm_v031_gdn_minimal.patch` (`+2.04 tok/s` lift, zero Python split overhead):** Replaces `core_attn_out = torch.zeros(...)` with `torch.empty(...)` on the fused C++ GDN decode path (saving 24 `cudaMemsetAsync` calls per step across the 24 GDN layers) and pre-compiles `causal_conv1d_update` during V1 startup profiling.
4. **`qwen35_d2g_v031.patch`:** Exposes `D2G_TILE=64`, `D2G_STAGES=2`, `D2G_WARPS=8` environment controls for the SM80 Triton `unified_attention` kernel used by the MTP draft head.

### Step 2 — Launch the `vLLM 0.31.0` Online HTTP Server
Use `serve_qwen35_v031_optimized.sh` (which automatically applies the 4 patches above if present in the same directory) or run the exact command below:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export D2G_TILE=64
export D2G_STAGES=2
export D2G_WARPS=8

exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8100 \
    --served-model-name maiprofile-qwen35-sft \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 25600 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --dtype bfloat16 \
    --trust-remote-code \
    --hf-overrides '{"architectures": ["Qwen3_5ForConditionalGeneration"]}' \
    --async-scheduling \
    --no-enable-log-requests \
    --disable-uvicorn-access-log \
    --mamba-ssm-cache-dtype bfloat16 \
    --no-enable-prefix-caching \
    --attention-backend FLASHINFER \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --language-model-only \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "TRITON_ATTN"}'
```

### Step 3 — Post-Boot Warmup & Client Request Contract
1. **8-Request Pre-Traffic Warmup:** Immediately after `/health` returns `200 OK`, send 8 concurrent short `/v1/chat/completions` requests (`max_tokens: 16, temperature: 0.0`) so Triton/FlashInfer JIT caches and CUDA graph dispatch tables are hot before production traffic arrives.
2. **Client Payload Contract (`/v1/chat/completions`):**
   Always pass `"temperature": 0.0, "top_p": 1.0` in the request payload (do not leave `"temperature": 0.2` in batch/evaluation JSONL rows):
   ```json
   {
     "model": "maiprofile-qwen35-sft",
     "messages": [{"role": "user", "content": "..."}],
     "max_tokens": 8192,
     "temperature": 0.0,
     "top_p": 1.0
   }
   ```

---

## 3. End-to-End Benchmark Reproduction (`3,879.50 output tok/s` on 2,000 Requests)

To guarantee that any engineer can independently reproduce the exact `3,879.50 output tok/s` result (as well as the `3,843.12 output tok/s` zero-patch greedy control), all model/dataset locations, artifact SHA-256 checksums, and standalone evaluation commands are documented below.

### 3.1 Model Checkpoint & Evaluation Dataset Provenance
- **Model Checkpoint (`Qwen3_5ForConditionalGeneration`, BF16, 24×GDN + 8×Full-Attention + 1-layer MTP):**
  - **AzureML / Singularity Datastore URI:** `azureml://subscriptions/e033d469-29f3-4d27-b736-322c44d8f206/resourcegroups/ranking/workspaces/aml-ranking-uksouth/datastores/adls_msn_dni_09_rankfun/paths/shares/users/zxy/autobench/data/qwen35-posted-reproduction-20261023-step6000/checkpoint`
  - **Hugging Face Repo:** `ChenWu/qwen35-l1234-v25-step6000`
  - **DLIS Falcon ModelDataPath:** `abfs://dtalake@copilotexplore.dfs.core.windows.net/MAI-Profile-Qwen35-SFT-step12000-20261006/cur_best_step_12000_ppl_1.2697`
- **2,000-Request Evaluation Dataset (`l1_v2_gpt56_test.jsonl`):**
  - **AzureML Datastore URI:** `azureml://datastores/adls_msn_dni_09_rankfun/paths/shares/users/zxy/autobench/data/qwen35-posted-reproduction-20261023-step6000/data/l1_v2_gpt56_test.jsonl`
  - **Dataset Invariants:** Exactly `2,000` JSONL rows, `5,871,663` total prompt tokens (`mean_input_tokens = 2,935.83`), `max_tokens = 8,192`.
  - **Critical Note on Row Temperature:** Each row in `l1_v2_gpt56_test.jsonl` contains `"temperature": 0.2`. For strict greedy reproduction (`3,879.50 tok/s` and `3,843.12 tok/s`), the evaluator overrides each request payload to `"temperature": 0.0, "top_p": 1.0` (the default behavior of `reproduce_qwen35_3879tps_eval.py`). Pass `--temperature -1` to preserve the dataset's `"temperature": 0.2` (`3,827.28 tok/s`).

### 3.2 GitHub Repositories & Artifact SHA-256 Checksums
All patches, serving launchers, standalone evaluation scripts, and AzureML submission pipelines are committed and pushed to GitHub (and hosted directly at `https://privacypshowcase12114.z13.web.core.windows.net/patches/`):
- **Benchmark & AML Submission Repo (`agentic_vllm`):** `https://github.com/xinyizou_microsoft/agentic_vllm.git` (branch `gemma4-optimization`, commit `72193fd`, directory `benchmarks/qwen35_posted_eval/` and `patches/attention/`)
- **Showcase & Standalone Patch Bundle Repo (`ensemble_privacy_pipeline`):** `https://github.com/overwindows/ensemble_privacy_pipeline.git` (branch `main`, directory `showcase/patches/`)

| Artifact Filename | SHA-256 Checksum | Role |
|---|---|---|
| `vllm_v031_eagle_full_cudagraph.patch` | `f10362f3a1ee475f9ee4aa5cf3095a404db90936424ddf42bf5662c6e5fff223` | Champion Patch 1 (`AutoRegressiveSpeculator` `FULL_DECODE_ONLY` CUDA graphs) |
| `vllm_v031_fast_spec_sample.patch` | `ea9fe30b0eec2c67c96a58407c306311262658fbb0844556da540c73c9f02c44` | Champion Patch 2 (BF16 zero-copy fused-temperature rejection & Gumbel sampler) |
| `vllm_v031_gdn_minimal.patch` | `9442b0056fd12dc1019e490412cb73eb94f529f6663652664f4ec3b0f0352042` | Champion Patch 3 (Fused C++ GDN `torch.empty` + `causal_conv1d_update` warmup) |
| `qwen35_d2g_v031.patch` | `0ae6eb05317cfd4909efe8c2720c2ef77ce99f23d70d0b0b507ea0fdd184239f` | Champion Patch 4 (`D2G_TILE=64`, `D2G_STAGES=2`, `D2G_WARPS=8` controls) |
| `serve_qwen35_v031_optimized.sh` | `66f119a945e670a30f453da00b6dcb7c966aa35c7f58d1e68445eac9db8dc26f` | Turnkey `vllm serve` launcher (auto-applies the 4 patches above) |
| `reproduce_qwen35_3879tps_eval.py` | `5c46cad4f05f7b31a4c2ab14b7717a680ae56ddef95325c69fd704898fe054da` | Standalone 2,000-request online HTTP `/v1/chat/completions` benchmark harness |

### 3.3 Option A — Turnkey Single-Command Reproduction on Any 1× A100-80GB Machine
Clone `agentic_vllm` (or download the 6 files from `https://privacypshowcase12114.z13.web.core.windows.net/patches/`) in an environment with `vllm==0.31.0` (`pip install vllm==0.31.0 aiohttp`) and run:

```bash
git clone -b gemma4-optimization https://github.com/xinyizou_microsoft/agentic_vllm.git
cd agentic_vllm/benchmarks/qwen35_posted_eval

# 1) Reproduce the 3,879.50 output tok/s Wave-6g 4-Patch Greedy Champion:
python3 reproduce_qwen35_3879tps_eval.py \
    --launch-server ./serve_qwen35_v031_optimized.sh \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH/l1_v2_gpt56_test.jsonl" \
    --output-dir ./repro_wave6g_champion_3879tps

# 2) Reproduce the 3,843.12 output tok/s Zero-Patch Stock Greedy Control:
VLLM_APPLY_PATCHES=false python3 reproduce_qwen35_3879tps_eval.py \
    --launch-server ./serve_qwen35_v031_optimized.sh \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH/l1_v2_gpt56_test.jsonl" \
    --output-dir ./repro_wave6f_stock_greedy_3843tps
```

`reproduce_qwen35_3879tps_eval.py` executes the exact benchmark protocol:
1. Launches `serve_qwen35_v031_optimized.sh` and polls `GET /health` + `GET /v1/models` until `maiprofile-qwen35-sft` is ready.
2. Runs **8 concurrent pre-timer HTTP warmup requests** (`max_tokens=16, temperature=0.0, top_p=1.0`).
3. Scrapes `GET /metrics` (`metrics_before.txt`), starts `time.perf_counter()`, dispatches all **2,000 `/v1/chat/completions` requests concurrently** (`concurrency=2000, max_tokens=8192, temperature=0.0, top_p=1.0`), stops the wall clock when the 2,000th response completes, and scrapes `GET /metrics` (`metrics_after.txt`).
4. Writes `summary.json` and verifies `successful_requests == 2000`, `prompt_tokens == 5871663`, `json_parseable == 2000`, `greedy_decoding_verified == true`, and `mtp_engagement_verified == true` (`draft_acceptance_rate >= 0.765`).

### 3.4 Option B — Reproduce on AzureML / Singularity (`aml-ranking-uksouth`)
To reproduce the exact AML jobs (`magenta_book_0g0hp7t2cy` / `frank_jelly_p13140lkhv` / `hungry_cat_jqbhqqvz0z`) from the committed `agentic_vllm` repository:

```bash
git clone -b gemma4-optimization https://github.com/xinyizou_microsoft/agentic_vllm.git
cd agentic_vllm
pip install -e .

python experiments/submit_qwen35_v031_http_online.py --submit \
    --only HTTP_V031_WAVE6G_GREEDY_MINIMAL_STACK_K3 \
           HTTP_V031_WAVE6G_GREEDY_EAGLE_FULLCG_SAMPLE_K3 \
           HTTP_V031_WAVE6F_GREEDY_R9_CTRL
```

---

## 4. DLIS `deployment-params.yml` Drop-In Snippet

```yaml
Operation: Deploy
Namespace: dlis-coreranker
ApplicationName: maiprofile-qwen35-sft-step12000-v1
DeployCluster: FalconCentralUS_APRegionalIsland
Replicas: 1
DeploymentParams:
  ModelPath: dlisfalconprodcontainerregistry.azurecr.io/maiprofile/qwen35-sft-vllm:step12000-v031-opt
  Dri: isstdri-wuc
  WorkingDir: /workspace
  ModelDataPath: abfs://dtalake@copilotexplore.dfs.core.windows.net/MAI-Profile-Qwen35-SFT-step12000-20261006/cur_best_step_12000_ppl_1.2697
  Commands:
    - /bin/bash
    - /mnt/model/shared/MAI-Profile-Qwen35-SFT-step12000-20261006/cur_best_step_12000_ppl_1.2697/serve_qwen35_v031_optimized.sh
    - "8100"
    - "1"
    - /mnt/model/shared/MAI-Profile-Qwen35-SFT-step12000-20261006/cur_best_step_12000_ppl_1.2697
  Envs:
    - VLLM_DRAFT_ATTENTION_BACKEND=TRITON_ATTN
    - VLLM_NUM_SPECULATIVE_TOKENS=3
    - VLLM_MAX_NUM_SEQS=128
    - VLLM_MAX_NUM_BATCHED_TOKENS=16384
    - VLLM_GPU_MEMORY_UTILIZATION=0.9
    - VLLM_ENABLE_PREFIX_CACHING=false
    - D2G_TILE=64
    - D2G_STAGES=2
    - D2G_WARPS=8
```

---

## 5. Startup & Runtime Verification Checklist

Check the server startup log and `/metrics` endpoint for these four invariants:
- `Using FLASHINFER attention backend` (Target 8 full-attention layers) and `TRITON_ATTN` on `model.mtp.layers.0.self_attn`.
- `Capturing CUDA graphs (decode, FULL_DECODE_ONLY)` for the `AutoRegressiveSpeculator` draft head alongside `(mixed prefill-decode, PIECEWISE)` for the target model.
- `KV cache configuration: 4 groups` (3× `MambaSpec` groups of 8 GDN layers + 1× `FullAttentionSpec` group of 9 attention layers = 8 target + 1 MTP draft).
- `/metrics` counters confirm `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total >= 0.765` (mean acceptance length `~3.30` tokens/step).
