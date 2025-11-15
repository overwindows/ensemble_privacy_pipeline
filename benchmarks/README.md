# Benchmarks

This folder contains various benchmark scripts for evaluating the privacy pipeline.

## Working Benchmarks

### ✅ `neutral_benchmark.py` ⭐ VENDOR-NEUTRAL FLAGSHIP!
- **Status**: Production-ready (part of comprehensive benchmark suite)
- **Purpose**: **Vendor-neutral synthetic benchmark** - the CLEANEST baseline for ensemble-redaction evaluation
- **Key Features**:
  - 🎯 **No Microsoft-specific field names** (unlike legacy demos)
  - 🏥 **Multi-domain coverage**: Medical, Financial, Education
  - 🔐 **Privacy-first design**: Tests ensemble consensus on masked PII
  - 📊 **Dual testing**: Privacy leakage + Utility preservation
- **Usage**:
  ```bash
  # Test all domains (300 samples = 100 medical + 100 financial + 100 education)
  python3 benchmarks/neutral_benchmark.py --benchmark all --domains all --num-samples 100

  # Test single domain
  python3 benchmarks/neutral_benchmark.py --benchmark privacy_leakage --domains medical --num-samples 50

  # Test utility only
  python3 benchmarks/neutral_benchmark.py --benchmark utility --domains all --num-samples 20
  ```
- **What it tests**:
  1. **Privacy Leakage Test**: Does the ensemble-redaction pipeline prevent PII exposure?
     - Generates synthetic user data with known PII (names, conditions, medications, etc.)
     - Applies **4-model ensemble** (gpt-oss-120b, DeepSeek-V3.1, Qwen3-32B, DeepSeek-V3-0324)
     - Checks if PII keywords appear in **consensus output** (median voting)
     - Reports: PII exposure rate, protection rate
  2. **Utility Preservation Test**: Does the pipeline maintain accuracy?
     - Tests topic matching accuracy
     - Verifies ensemble maintains 85%+ accuracy despite PII masking
- **Domains**:
  - **Medical**: Health conditions, medications, doctor names, hospital visits
  - **Financial**: Bank accounts, credit cards, transaction amounts, institutions
  - **Education**: Schools, GPAs, majors, scholarships, professors
- **Field Format**: Uses **vendor-neutral** schema
  - `raw_queries`: Generic search queries/prompts
  - `browsing_history`: Web pages viewed
  - `demographics`: Age, gender, location (optional)
  - ❌ No `MSNClicks`, `BingSearch`, or `MAI` (those are in legacy demos only)
- **Why vendor-neutral?**
  - Aligns with public dataset formats (ai4privacy, PUPA, TAB)
  - Generalizable to any LLM API (not Microsoft-specific)
  - Academic publication-ready
- **Cost**: ~$40-50 for 300 samples (1,200 API calls = 300 samples × 4 models)
- **Output**: `neutral_benchmark_results.json`
- **Estimated time**: 60-75 minutes for 300 samples

### ✅ `comparison.py`
- **Status**: Production-ready (imports fixed)
- **Purpose**: Compare baseline vs privacy-preserving approaches
- **Usage**: Integrated into `../run_benchmarks.py`

### ✅ `public_datasets_simple.py` ⭐ NEW!
- **Status**: Production-ready
- **Purpose**: Evaluate on real **ai4privacy/pii-masking-200k** dataset (200K+ samples, 54 PII types)
- **Usage**:
  ```bash
  pip install datasets huggingface_hub
  python3 benchmarks/public_datasets_simple.py --num-samples 100
  ```
- **Tests**: Real PII detection (emails, names, addresses, phone numbers, etc.)

### ✅ `dp_benchmark.py` ⭐ NEW!
- **Status**: Production-ready
- **Purpose**: Compare your ensemble approach with **Differential Privacy (DP)**
- **Usage**:
  ```bash
  python3 benchmarks/dp_benchmark.py --num-samples 20
  ```
- **Tests**:
  - **Canary Exposure** (PrivLM-Bench style): Can attackers extract unique strings?
  - **Membership Inference Attack (MIA)**: Can attackers determine if data was used?
  - **DP Comparison**: How does your approach compare to ε=1.0 and ε=5.0 DP?
- **Cost**: ~$2-3 for 20 samples (tests 3 scenarios)

### ✅ `pupa_benchmark.py` ⭐ NEW!
- **Status**: Production-ready
- **Purpose**: Evaluate on **PUPA (Private User Prompt Annotations)** dataset from NAACL 2025
- **Paper**: "PAPILLON: Privacy Preservation from Internet-based and Local Language Model Ensembles" (Li et al., NAACL 2025)
- **Dataset**: 901 real-world user-agent interactions from WildChat corpus
- **Usage**:
  ```bash
  # With real PUPA dataset (if available)
  python3 benchmarks/pupa_benchmark.py --dataset-path /path/to/pupa.json --num-samples 100

  # With simulated WildChat-style data
  python3 benchmarks/pupa_benchmark.py --simulate --num-samples 50
  ```
- **Tests**:
  - **Privacy Leakage**: % of PII units exposed in output
  - **Quality Preservation**: Response quality maintenance
  - **Category Analysis**: Performance on 3 PII categories:
    1. Job, Visa, & Other Applications
    2. Financial and Corporate Info
    3. Quoted Emails and Messages
- **Baseline**: PAPILLON achieved 85.5% quality with 7.5% privacy leakage
- **Dataset Source**: https://github.com/Columbia-NLP-Lab/PAPILLON
- **Cost**: ~$2-4 for 50 samples

### ✅ `text_sanitization_benchmark.py` ⭐ NEW!
- **Status**: Production-ready
- **Purpose**: Evaluate on **TAB (Text Anonymization Benchmark)** - ECHR court cases
- **Paper**: Pilán et al. (2022) "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization"
- **Dataset**: 1,268 English-language court cases from European Court of Human Rights
- **Usage**:
  ```bash
  # With real TAB dataset (if available)
  git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark
  python3 benchmarks/text_sanitization_benchmark.py --dataset-path text-anonymization-benchmark/data --num-samples 100

  # With simulated ECHR-style data
  python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 50
  ```
- **Tests**:
  - **PII Masking Rate**: % of PII entities successfully masked
  - **Direct ID Protection**: Protection of direct identifiers (PERSON, etc.)
  - **Quasi ID Protection**: Protection of quasi-identifiers (LOC, DATETIME, etc.)
  - **Entity Type Breakdown**: Performance by entity type (PERSON, ORG, LOC, DATETIME, CODE, etc.)
- **Baselines Compared**:
  - Naive Redaction: 65% masking
  - SanText (DP-based): 82% masking
  - State-of-the-art: 92% masking
- **Dataset Source**: https://github.com/NorskRegnesentral/text-anonymization-benchmark
- **Cost**: ~$2-4 for 50 samples

## Experimental Benchmarks (Require Additional Setup)

### ⚠️ `public_datasets.py`
- **Status**: Requires `evaluation_framework` module (not included)
- **Purpose**: Test on public datasets (ai4privacy, PII-Bench, PrivacyXray)
- **Dependencies**:
  - `evaluation_framework` module (needs to be created)
  - `pip install datasets huggingface_hub`
- **Note**: For production use, run `../run_benchmarks.py` instead

### ⚠️ `dp_specific.py`
- **Status**: Requires `evaluation_framework` module (not included)
- **Purpose**: Differential Privacy comparisons (canary exposure, MIA, NIST)
- **Dependencies**: `evaluation_framework` module (needs to be created)
- **Note**: Contains theoretical DP comparison code

## Recommended Usage

For immediate, production-ready benchmarking:

```bash
# From repository root
python3 run_benchmarks.py --benchmark all --num-samples 20
```

This unified benchmark runner:
- ✅ Works out of the box
- ✅ Uses your configured LLM API models
- ✅ Tests privacy leakage, utility, and baseline comparison
- ✅ Generates JSON results report

## Future Work

To make `public_datasets.py` and `dp_specific.py` work:

1. Create `evaluation_framework.py` module with:
   - `EvaluationPipeline`
   - `PrivacyEvaluator`
   - `UtilityEvaluator`
   - `BenchmarkDatasetLoader`

2. Update imports to use `src.privacy_core` instead of mock code

3. Integrate with real LLM APIs
