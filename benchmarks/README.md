# Benchmarks

This folder contains various benchmark scripts for evaluating the privacy pipeline.

## Working Benchmarks

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
- ✅ Uses your real SambaNova models
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
