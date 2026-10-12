# Reproducing the Best Gemma-4-26B-A4B Throughput Result (S2, ~2403 output TPS)

Standalone recipe for anyone outside the campaign to reproduce the current throughput
champion on a single **NVIDIA A100 80GB (sm_80)**:

> **2403.18 output tokens/s** (QPS 1.80, 556.66s per 1000 prompts at fixed output_len 8192)
> — the `feat/gemma4-decode2d-geom` "S2" geometry, **reproducibly re-confirmed at 2402.49**.

This is the **throughput** result only. It is **not** the quality-gated int4 target (that
lever to ~2500 is still mid quality-check; see the live page).

> **To reproduce the 2403 throughput you need ONLY the vllm-msn fork below.** The fork
> (branch `feat/gemma4-decode2d-geom` @ `a3d159271`) is committed and pushed; the recipe
> in §2 is the exact config that ran. The AML harness/config YAML (`agentic_vllm`,
> `gemma4-s2-ctrl-unified-ab-a100.yaml`) is a convenience wrapper — included for
> fidelity, but the numbers come from the fork + env/geometry + MTP wiring, all of which
> are spelled out here so you can re-derive them without the harness if you prefer.

---

## 1. The implementation — a dedicated fork branch

- **Fork repo:** `https://github.com/wuc_microsoft/vllm-msn` (private fork of `MS-Feeds/vllm-msn`)
- **Branch:** `feat/gemma4-decode2d-geom`

Use the **baked** commit (the one the champion run actually executed, and the current
branch head):

| SHA | Note |
|-----|------|
| **`a3d159271`** | **Reproduce this.** "Bake S2 decode-2D geometry (D2G_TILE=32, D2G_STAGES=2) as fork default". Changes `vllm/envs.py`. Because S2 is baked, a plain branch clone runs S2 with **no env plumbing**. |
| `605c39c7a` | ROUND-25 S2 winner (pre-bake; env-override form of the same geometry). |

```bash
git clone https://github.com/wuc_microsoft/vllm-msn.git
git checkout feat/gemma4-decode2d-geom    # head = a3d159271 (S2 baked as default)
```

The bake is a single, small change to `vllm/envs.py` — very portable to cherry-pick:

```
# vllm/envs.py — S2 defaults: tile=32, stages=2, warps=0
# D2G_* = 0 reverts to stock prefill-tuned behavior
D2G_TILE=32, D2G_STAGES=2, D2G_WARPS=0  (baked defaults)
```

---

## 2. The run harness

Reproduce with the ROUND-27b unified A/B config (this gives the **apples-to-apples**
TPS + QPS + latency numbers; earlier rounds only had comparable output TPS):

- **Config:** `experiments/gemma4-s2-ctrl-unified-ab-a100.yaml` in the **harness repo**
  [`xinyizou_microsoft/agentic_vllm`](https://github.com/xinyizou_microsoft/agentic_vllm) @ branch
  `gemma4-optimization` (commit `f9b77b3`). Raw:
  `https://raw.githubusercontent.com/xinyizou_microsoft/agentic_vllm/gemma4-optimization/experiments/gemma4-s2-ctrl-unified-ab-a100.yaml`
- **Install mode:** `vllm_install_mode: source` (so the fork change is actually compiled in)
- **Dataset:** `sc1_delta_v2.jsonl`, **1000 prompts**, **output_len 8192**, greedy
  (`temperature: 0.0`, `top_p: 1.0`).

### The S2 (champion) arm
```yaml
s2:
  vllm_repo: https://github.com/wuc_microsoft/vllm-msn.git
  vllm_branch: feat/gemma4-decode2d-geom     # @ a3d159271
  env_vars:
    VLLM_TEST_FORCE_FP8_MARLIN: "1"     # from hardware.fixed_env_vars
    D2G_TILE: "32"
    D2G_STAGES: "2"
    D2G_WARPS: "0"
    # fixed_env_vars (hardware — every arm gets these):
    VLLM_USE_FLASHINFER_MOE_FP8: "0"
    VLLM_USE_FLASHINFER_SAMPLER: "0"
    VLLM_MOE_USE_DEEP_GEMM: "0"
  config:
    quantization: fp8
    kv_cache_dtype: auto
    gpu_memory_utilization: 0.95
    moe_backend: marlin
    speculative_method: mtp
    spec_tokens: 5
    draft_model_subdir: assistant
    max_tokens: 8192
    enforce_eager: false
    max_num_seqs: 128
    max_num_batched_tokens: 16384
    max_model_len: 24576
    enable_chunked_prefill: true
    enable_prefix_caching: true
```

The runtime **injects the MTP draft** at submit time: it mounts the model store, sets
`speculative_config.method=mtp`, `num_speculative_tokens=5`, and points the draft at
`<model_store>/assistant/` (the `assistant/` draft subdir inside the Gemma4 deploy
checkpoint). Keep that `draft_model_subdir: assistant` wiring — it is load-bearing for
the MTP-k5 spec-decode speedup.

**Install-time env (must be replicated when building from source so the cutlass FP8 kernel
that is NOT used on sm_80 is disabled); these come from `submit.aml.install_env_vars`:**
```yaml
install_env_vars:
  TORCH_CUDA_ARCH_LIST: "8.0"
  MAX_JOBS: "8"
  NVCC_THREADS: "1"
  VLLM_USE_PRECOMPILED: "0"
  TORCH_COMPILE_DISABLE: "0"
  VLLM_DISABLED_KERNELS: "CutlassFP8ScaledMMLinearKernel"   # sm_80 fp8 path disabled; Marlin handles MoE
```

### The control arm (for a clean A/B)
The **control MUST explicitly pin `D2G_*=0`** — because S2 is now baked as the fork
default, a bare clone runs S2 unless you disable it:
```yaml
ctrl:
  env_vars:
    VLLM_TEST_FORCE_FP8_MARLIN: "1"     # + all fixed_env_vars above
    D2G_TILE: "0"
    D2G_STAGES: "0"
    D2G_WARPS: "0"
```

---

## 3. Hardware / environment

| Item | Value |
|------|-------|
| GPU | **NVIDIA A100 80GB**, compute capability `sm_80` (Ampere) |
| Instance | `Singularity.ND12am_A100_v4` |
| Quantization | `fp8` + **Marlin** MoE backend (this is the weight-HBM ceiling) |
| MTP draft | self-spec, 5 spec tokens, draft weights in `assistant/` subdir |
| gpu_memory_utilization | 0.95 |

---

## 4. Reference results (same unified harness, ROUND-27b)

| Arm | Job | Output TPS | QPS | 1000-pr elapsed |
|-----|-----|-----------|-----|-----------------|
| CTRL (D2G=0) | `icy_carnival_yxtffhqy10` | 2058.05 | 1.56 | 641.09s |
| **S2 (tile32/stg2)** | `teal_evening_xjzxx6yb99` | **2403.18** | **1.80** | **556.66s** |

Repro re-run confirmed **2402.49** — rock-solid.

---

## 5. Caveats for anyone reproducing

- **Comparability.** Output TPS is comparable across all rounds; **QPS / elapsed are only
  apples-to-apples within this unified 1000×8192 script row** (same dataset / output_len /
  commit).
- **~2403–2429 is the weight-HBM ceiling.** S2's decode-2D geometry cut the attention
  share; the remaining ~53% of step time is Marlin GEMM weight fetches, which this recipe
  does not touch. The **int4 W4A16 Humming** arm (halving weight bytes) is the lever past
  this — and is gated behind a quality A/B before it is recommended over S2.
- **Token-length caveat (ROUND-27b addendum).** Output-length means across arms were
  1305.4 (2055 record) / 1319.4 (CTRL) / 1337.8 (S2). On the fixed-8192 basis S2 generates
  slightly more tokens/prompt than the 2055 record, so part of the TPS gap is a higher
  token-generating basis, not pure speedup. Treat cross-basis (non-unified-harness)
  comparisons accordingly.

---

*Campaign log: `experiments/EXPERIMENT_LOG.md` in the same harness repo (ROUND-25/26/27b/28).
Live status: https://privacypshowcase12114.z19.web.core.windows.net/vllm-gemma4.html*
