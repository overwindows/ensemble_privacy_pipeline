# Ensemble-Redaction Privacy Pipeline Documentation

## Overview

The Ensemble-Redaction Privacy Pipeline provides privacy-preserving interest evaluation using a 4-step approach:
1. **Redaction & Masking** - Replace PII with anonymized tokens
2. **Split Inference** (optional) - Process inside privacy boundary
3. **Ensemble LLM Evaluation** - Multiple models independently evaluate masked data
4. **Consensus Aggregation** - Combine outputs using majority voting or median aggregation

**Key Benefits:**
- ✅ Removes PII via masking
- ✅ Suppresses rare details via ensemble voting
- ✅ Provides readable, useful outputs
- ✅ Works without training (inference-only)
- ⚠️ No formal mathematical guarantees (heuristic privacy)
- ⚠️ Requires 3-5x compute vs single model

---

## ⚠️ Critical Bugfix Notice (2025-01-19)

**Previous benchmark results are INVALID** due to a critical bug that was fixed.

**The Bug:**
- All public dataset benchmarks created 4 LLM evaluators but **only used the first model's output**
- This bypassed ensemble consensus entirely
- Previous results represented **single-model performance**, not ensemble

**The Fix:**
- All benchmarks now properly use ensemble consensus
- Uses **majority voting** (no ground truth during selection)
- Ground truth is only used for **measurement**, not **selection**

**Action Required:** All benchmarks must be **re-run** with corrected code to get valid ensemble performance numbers.

---

## Pipeline Overview

### How It Works

**Example Scenario:** Evaluating user interest in topics based on behavioral data.

**Step 1: Redaction**
- Original: `{"query": "diabetes diet plan", "timestamp": "2024-01-15T11:00:00"}`
- Masked: `{"token": "QUERY_SEARCH_001", "timestamp": "recent"}`
- Privacy: No specific queries exposed, but signal preserved

**Step 2: Ensemble Evaluation**
- 4-5 different LLM models independently score topics
- Each model produces: `{"ItemId": "A", "QualityScore": 0.85, "QualityReason": "VeryStrong:MSNUpvotes+MSNClicks"}`

**Step 3: Consensus Aggregation**
- **Text tasks:** Majority voting (≥50% models agree) or shortest text fallback
- **Structured tasks:** Median aggregation on scores, majority voting on reasons

**What Exits Privacy Boundary:**
- ✅ Only consensus JSON (ItemId, QualityScore, generic source types)
- ❌ NO specific queries, titles, URLs, timestamps, or exact demographics

---

## Benchmarks

### Available Benchmarks

#### 1. Vendor-Neutral Synthetic (`neutral_benchmark.py`)
**Purpose:** Clean baseline for ensemble-redaction evaluation

```bash
python3 benchmarks/neutral_benchmark.py --benchmark all --domains all --num-samples 100
```

**Tests:**
- Privacy leakage (PII exposure rate)
- Utility preservation (topic matching accuracy)

**Features:**
- Multi-domain: Medical, Financial, Education
- 4-model ensemble: gpt-oss-120b, DeepSeek-V3.1, Qwen3-32B, DeepSeek-V3-0324

---

#### 2. ai4privacy PII Masking (`public_datasets_simple.py`)
**Purpose:** Real PII dataset evaluation

```bash
pip install datasets huggingface_hub
python3 benchmarks/public_datasets_simple.py --num-samples 1000
```

**Dataset:** [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) (200K+ samples, 54 PII types)

---

#### 3. PUPA Question Answering (`pupa_benchmark.py`)
**Purpose:** Privacy-preserving question answering

```bash
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 901
```

**Paper:** Li et al., "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles", NAACL 2025  
**Baseline:** PAPILLON achieved 85.5% quality with 7.5% privacy leakage

---

#### 4. TAB Document Sanitization (`text_sanitization_benchmark.py`)
**Purpose:** Text anonymization benchmark

```bash
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 1268
```

**Paper:** Pilán et al. (2022) "The Text Anonymization Benchmark (TAB)", ACL 2022 Findings  
**Tests:** PII Masking Rate, Direct ID Protection, Quasi ID Protection

---

#### 5. Differential Privacy Comparison (`dp_benchmark.py`)
**Purpose:** Compare ensemble approach with formal DP

```bash
python3 benchmarks/dp_benchmark.py --num-samples 100
```

**Tests:**
- Canary Exposure (PrivLM-Bench style)
- Membership Inference Attack (MIA)
- DP Comparison (ε=1.0, ε=5.0)

---

### Running All Benchmarks

Use the wrapper script to run all benchmarks sequentially:

```bash
export LLM_API_KEY='your-api-key-here'
python3 run_all_benchmarks.py
```

This runs:
- Vendor-Neutral: 300 samples
- ai4privacy: 1,000 samples
- PUPA: 901 samples
- TAB: 1,268 samples
- DP Comparison: 100 samples

**Total:** 3,569 samples across 5 benchmarks

**Output:**
- Results: `results/*.json`
- Logs: `logs/*.log`
- Summary: `benchmark_suite_summary.json`

---

## Setup

### Prerequisites

```bash
# Set LLM API key
export LLM_API_KEY='your-api-key-here'

# Install dependencies (for ai4privacy benchmark)
pip install datasets huggingface_hub
```

### Directory Structure

```
results/          # Benchmark results (JSON files)
logs/             # Execution logs
benchmarks/       # Individual benchmark scripts
```

Directories are auto-created by benchmarks.

---

## Running Benchmarks

### Quick Validation Test (Recommended First)

Verify fixes work before full runs:

```bash
# 10 samples each (~$5)
python3 benchmarks/public_datasets_simple.py --num-samples 10
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 10
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 10
```

**What to check:**
- ✅ Console shows: `📊 Ensemble (4 models): X unique responses, consensus: majority`
- ✅ Consensus types: unanimous, majority, or shortest_fallback
- ✅ No errors/crashes

### Medium-Scale Test

```bash
# 100 samples each (~$50, ~2-3 hours)
python3 benchmarks/public_datasets_simple.py --num-samples 100
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 100
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 100
```

### Full-Scale Benchmarks

```bash
# Full scale (~$150-200, ~8-12 hours)
python3 benchmarks/public_datasets_simple.py --num-samples 1000
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 901
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 1268
```

---

## Expected Consensus Behavior

When running benchmarks, you'll see consensus messages:

**Strong consensus:**
```
📊 Ensemble (4 models): 2 unique responses, consensus: majority
✅ ALL PII PROTECTED (4 units)
```

**No consensus:**
```
📊 Ensemble (4 models): 4 unique responses, consensus: shortest_fallback
❌ PII LEAKED: 1/4 units
```

**Unanimous:**
```
📊 Ensemble (4 models): 1 unique response, consensus: unanimous
```

---

## Consensus Strategies

### Text Tasks (PUPA, Text Masking, TAB)
1. **Majority voting** - Select output agreed upon by ≥50% models
2. **Shortest text fallback** - If no majority, pick shortest response (heuristic: less text = safer)

### Structured Tasks (Interest Evaluation)
1. **Median aggregation** - Take median of numeric scores
2. **Majority voting** - Take most common reason across models

**Key Principle:** Ground truth is used for **MEASUREMENT**, not **SELECTION**.

---

## Privacy vs Utility Tradeoffs

| Aspect | Formal DP | This Pipeline |
|--------|-----------|---------------|
| **Noise mechanism** | Calibrated Gaussian/Laplace | Ensemble variance (natural noise) |
| **Privacy guarantee** | Mathematical (ε,δ)-DP | Heuristic (k-anonymity-like) |
| **Utility** | Can be poor for text | Good (readable outputs) |
| **Computation** | 1x LLM call + noise | 4-5x LLM calls |
| **Training required** | Yes (DP-SGD) | No (inference-only) |

**When to Use This Approach:**
- ✅ Can't train models (API-only access)
- ✅ Need readable outputs (not gibberish)
- ✅ Have compute budget for ensemble
- ✅ Accept heuristic privacy (not formal DP)

**Not a good fit when:**
- ❌ Need formal DP guarantees (regulatory requirement)
- ❌ Need single model (cost constraints)
- ❌ Generating long-form text (not scoring)

---

## References

1. **ai4privacy/pii-masking-200k**: https://huggingface.co/datasets/ai4privacy/pii-masking-200k
2. **PUPA Dataset**: Li et al., "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles", NAACL 2025
3. **TAB Dataset**: Pilán et al., "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization", ACL 2022 Findings
4. **PAPILLON Baseline**: Li et al., NAACL 2025 (Quality=85.5%, Leakage=7.5%)

---

## Troubleshooting

### Benchmarks Not Showing Ensemble Consensus

**Check:**
- Console should show `📊 Ensemble (4 models): ...` messages
- If missing, verify you're using the fixed code (after 2025-01-19)

### High API Costs

**Options:**
- Start with quick validation test (10 samples)
- Run benchmarks individually instead of all at once
- Check API pricing and rate limits

### Results Not Improving

**Possible reasons:**
- High model correlation (all models produce similar outputs)
- Detection accuracy issues (PII checker false negatives)
- Task mismatch (free-form text hard to aggregate)

**What to do:**
- Analyze per-sample results
- Check consensus distribution
- Consider different model combinations

---

## Summary

This pipeline provides **practical privacy** without formal guarantees, with better utility than DP text generation. It's a pragmatic solution when:
- Formal DP is too costly (gibberish outputs)
- You still need strong practical privacy
- You have compute budget for ensemble

**Next Steps:**
1. Run quick validation test (10 samples)
2. If successful, proceed to medium or full-scale benchmarks
3. Compare results with single-model baseline
4. Document findings and update results
