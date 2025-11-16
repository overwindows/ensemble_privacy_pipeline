# Benchmark Evaluation

## Public Benchmark Results

| Benchmark | Samples | Metric | Proposed Approach | Baseline | Difference |
|-----------|---------|--------|-------------------|----------|------------|
| **[ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)** | 1,000 | Full Protection | 28.8% (288/1000) | - | - |
| | | PII Types Tested | 54 types | - | - |
| **PUPA** (Li et al., NAACL 2025) | 901 | Response Success | 100.0% (901/901) | 85.5% | +14.5% |
| | | Privacy Leakage | 18.8% (902/4806) | 7.5% | +11.3% |
| **TAB** (Pilán et al., ACL 2022) | 1,268 | Direct ID Protection | 99.9% (1267/1268) | - | - |
| | | Quasi ID Protection | 99.9% (3801/3804) | - | - |
| | | Overall PII Masking | 83.7% (5308/6340) | - | - |

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
