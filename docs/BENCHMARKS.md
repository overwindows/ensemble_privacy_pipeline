# Benchmark Evaluation

## ⚠️ IMPORTANT: Previous Results Invalid - Bugfix Applied (2025-01-19)

**Critical bug discovered:** All public dataset benchmarks (Text Masking, PUPA, TAB) were creating 4 LLM evaluators but **only using the first model's output**, completely bypassing ensemble consensus.

**Impact:**
- ❌ Previous benchmark results represented **single-model performance** (gpt-oss-120b only)
- ❌ 75% of API costs were wasted (paid for 4 models, used 1)
- ❌ All comparisons claiming "ensemble improves privacy" were unsupported

**Fix applied:** All benchmarks now properly use ensemble consensus (selecting output with **lowest PII leakage**).

**Action required:** All benchmarks must be **re-run** with corrected code to get valid ensemble performance numbers.

See [Critical Bugfix Details](#critical-bugfix-ensemble-consensus-2025-01-19) below.

---

## Public Benchmark Results (⚠️ OUTDATED - NEEDS RE-RUN)

**These numbers are from the BROKEN code (single model only, NOT ensemble):**

| Benchmark | Samples | Metric | OLD (Single Model) | Baseline | Status |
|-----------|---------|--------|-------------------|----------|---------|
| **[ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)** | 1,000 | Full Protection | 28.8% (288/1000) | - | ⚠️ **INVALID** |
| | | PII Types Tested | 54 types | - | - |
| **PUPA** (Li et al., NAACL 2025) | 901 | Response Success | 100.0% (901/901) | 85.5% | ⚠️ **INVALID** |
| | | Privacy Leakage | 18.8% (902/4806) | 7.5% | ⚠️ **INVALID** |
| **TAB** (Pilán et al., ACL 2022) | 1,268 | Direct ID Protection | 99.9% (1267/1268) | - | ⚠️ **INVALID** |
| | | Quasi ID Protection | 99.9% (3801/3804) | - | ⚠️ **INVALID** |
| | | Overall PII Masking | 83.7% (5308/6340) | - | ⚠️ **INVALID** |

**Valid results:**
- ✅ Vendor-Neutral Synthetic benchmark (300 samples) - Actually used ensemble
- ✅ DP Comparison benchmark (100 samples) - Actually used ensemble

---

## Available Benchmarks

### Public Dataset Benchmarks

#### `public_datasets_simple.py` - ai4privacy PII Masking
Evaluate on real **ai4privacy/pii-masking-200k** dataset (200K+ samples, 54 PII types).

```bash
pip install datasets huggingface_hub
python3 benchmarks/public_datasets_simple.py --num-samples 100
```

**Dataset**: https://huggingface.co/datasets/ai4privacy/pii-masking-200k

---

#### `pupa_benchmark.py` - PUPA Question Answering
Evaluate on **PUPA (Private User Prompt Annotations)** dataset from NAACL 2025.

```bash
# With simulated WildChat-style data
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 50

# With real PUPA dataset (if available)
python3 benchmarks/pupa_benchmark.py --dataset-path /path/to/pupa.json --num-samples 100
```

**Paper**: Li et al., "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles", NAACL 2025
**Dataset**: https://github.com/Columbia-NLP-Lab/PAPILLON
**Tests**: Privacy Leakage, Quality Preservation, Category Analysis
**Baseline**: PAPILLON achieved 85.5% quality with 7.5% privacy leakage

---

#### `text_sanitization_benchmark.py` - TAB Document Sanitization
Evaluate on **TAB (Text Anonymization Benchmark)** - ECHR court cases.

```bash
# With simulated ECHR-style data
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 50

# With real TAB dataset (if available)
git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark
python3 benchmarks/text_sanitization_benchmark.py --dataset-path text-anonymization-benchmark/data --num-samples 100
```

**Paper**: Pilán et al. (2022) "The Text Anonymization Benchmark (TAB)", ACL 2022 Findings
**Dataset**: https://github.com/NorskRegnesentral/text-anonymization-benchmark
**Tests**: PII Masking Rate, Direct ID Protection, Quasi ID Protection, Entity Type Breakdown

---

### Privacy Attack Benchmarks

#### `dp_benchmark.py` - Differential Privacy Comparison
Compare ensemble approach with **Differential Privacy (DP)**.

```bash
python3 benchmarks/dp_benchmark.py --num-samples 20
```

**Tests**:
- **Canary Exposure** (PrivLM-Bench style): Can attackers extract unique strings?
- **Membership Inference Attack (MIA)**: Can attackers determine if data was used?
- **DP Comparison**: How does your approach compare to ε=1.0 and ε=5.0 DP?

**Cost**: ~$2-3 for 20 samples

---

### Synthetic Benchmarks

#### `neutral_benchmark.py` - Vendor-Neutral Flagship
**Vendor-neutral synthetic benchmark** - the cleanest baseline for ensemble-redaction evaluation.

```bash
# Test all domains (300 samples = 100 medical + 100 financial + 100 education)
python3 benchmarks/neutral_benchmark.py --benchmark all --domains all --num-samples 100

# Test single domain
python3 benchmarks/neutral_benchmark.py --benchmark privacy_leakage --domains medical --num-samples 50

# Test utility only
python3 benchmarks/neutral_benchmark.py --benchmark utility --domains all --num-samples 20
```

**Key Features**:
- 🎯 No Microsoft-specific field names (generic schema)
- 🏥 Multi-domain coverage: Medical, Financial, Education
- 🔐 Privacy-first design: Tests ensemble consensus on masked PII
- 📊 Dual testing: Privacy leakage + Utility preservation

**What it tests**:
1. **Privacy Leakage Test**: Does ensemble-redaction prevent PII exposure?
   - Applies 4-model ensemble (gpt-oss-120b, DeepSeek-V3.1, Qwen3-32B, DeepSeek-V3-0324)
   - Checks if PII keywords appear in consensus output (median voting)
   - Reports: PII exposure rate, protection rate

2. **Utility Preservation Test**: Does the pipeline maintain accuracy?
   - Tests topic matching accuracy
   - Verifies ensemble maintains 85%+ accuracy despite PII masking

**Cost**: ~$40-50 for 300 samples (1,200 API calls = 300 samples × 4 models)
**Output**: `results/neutral_benchmark_results.json`
**Estimated time**: 60-75 minutes for 300 samples

---

## Setup

All benchmarks require:

```bash
# Set LLM API key
export LLM_API_KEY='your-api-key-here'

# Ensure results and logs directories exist (auto-created by benchmarks)
# Results saved to: results/
# Logs saved to: logs/
```

---

## References

1. **ai4privacy/pii-masking-200k**: https://huggingface.co/datasets/ai4privacy/pii-masking-200k
2. **PUPA Dataset**: Li et al., "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles", NAACL 2025
3. **TAB Dataset**: Pilán et al., "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization", ACL 2022 Findings
4. **PAPILLON Baseline**: Li et al., NAACL 2025 (Quality=85.5%, Leakage=7.5%)

---

## Critical Bugfix: Ensemble Consensus (2025-01-19)

### Problem 1: Not Using All Models

All public dataset benchmarks created 4 LLM evaluators but only used the first one's output.

### Problem 2: Using Ground Truth for Selection (The Critical Insight)

The initial fix attempted to use ground truth PII labels to SELECT the best output. This is **scientifically invalid** because:

**Invalid approach (circular logic):**
```python
# WRONG: Uses ground truth during selection
for output in outputs:
    leakage = check_pii_leakage(output, ground_truth_pii)  # ← Cheating!
best = min(outputs, key=lambda x: x.leaked_count)
```

**Why wrong:** If you already know the ground truth, why do you need ensemble at all?

### Proper Fix: Ground-Truth-Free Aggregation

Proper evaluation requires two separate phases:
1. **Aggregate outputs WITHOUT ground truth** (simulates production)
2. **Then evaluate aggregated output WITH ground truth** (measures quality)

**Correct Pattern:**
```python
# Phase 1: Aggregate WITHOUT using ground truth
from collections import Counter
output_counts = Counter(outputs)

if output_counts.most_common(1)[0][1] >= len(outputs) / 2:
    final = output_counts.most_common(1)[0][0]  # Majority consensus
else:
    final = min(outputs, key=len)  # Fallback: shortest (safer)

# Phase 2: THEN evaluate (not used for selection!)
leakage = check_pii_leakage(final, ground_truth_pii)
```

**Key principle:** Ground truth is used for **MEASUREMENT**, not **SELECTION**.

### Consensus Strategies by Task Type

**Text tasks (PUPA, Text Masking, TAB):**
1. Majority voting (≥50% models agree)
2. Shortest text fallback (heuristic: less text = safer)

**Structured tasks (Interest Evaluation):**
1. Median aggregation on numeric scores
2. Majority voting on categorical reasons

### What to Expect

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

### Files Modified

1. [benchmarks/pupa_benchmark.py:304-331](../benchmarks/pupa_benchmark.py#L304-L331)
2. [benchmarks/public_datasets_simple.py:159-183](../benchmarks/public_datasets_simple.py#L159-L183)
3. [benchmarks/text_sanitization_benchmark.py:275-299](../benchmarks/text_sanitization_benchmark.py#L275-L299)

### Verification

Test the fix without running full benchmarks:
```bash
python src/verify_fix.py
```

This demonstrates the difference between invalid (ground-truth selection) and valid (majority voting) approaches.

### Run Corrected Benchmarks

**Quick test (10 samples):**
```bash
set LLM_API_KEY=your-key-here
python tests/run_test_benchmarks.py
```

**Full benchmarks:**
```bash
set LLM_API_KEY=your-key-here
python run_all_benchmarks.py
```
