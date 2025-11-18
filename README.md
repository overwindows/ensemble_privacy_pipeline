# Ensemble-Redaction Consensus Pipeline

**A Training-Free Privacy-Preserving Approach for LLM Inference on Sensitive User Data**

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Benchmarks and Results](#benchmarks-and-results)
- [Comparison with Differential Privacy](#comparison-with-differential-privacy)
- [References](#references)

## Overview

This repository implements a privacy-preserving pipeline for LLM inference that combines input masking with ensemble consensus voting. The approach is designed to reduce PII exposure when processing sensitive user data through large language models.

**Features:**
- Training-free implementation compatible with any LLM API
- Input masking layer to redact PII before model inference
- Ensemble consensus to reduce individual model variance
- Evaluation on public privacy benchmarks

## Installation

```bash
# Clone the repository
git clone https://github.com/overwindows/ensemble-privacy-pipeline.git
cd ensemble-privacy-pipeline

# Install dependencies
pip3 install -r requirements.txt

# Set your LLM API key (OpenAI-compatible API)
export LLM_API_KEY='your-key-here'
```

## Quick Start

Run the demonstration pipeline:

```bash
python3 run_demo_pipeline.py
```

Run the benchmark suite:

```bash
python3 run_all_benchmarks.py
```

## Usage

```python
from src.privacy_core import PrivacyRedactor, ConsensusAggregator
from examples.real_llm_example import RealLLMEvaluator
import os

# 1. Redact sensitive data
redactor = PrivacyRedactor()
masked_data = redactor.redact_user_data(raw_user_data)

# 2. Evaluate with ensemble models
api_key = os.getenv("LLM_API_KEY")
ensemble_models = ["model-1", "model-2", "model-3", "model-4"]
all_results = [
    RealLLMEvaluator(m, api_key).evaluate_interest(masked_data, topics)
    for m in ensemble_models
]

# 3. Aggregate with consensus
aggregator = ConsensusAggregator()
final_output = aggregator.aggregate_median(all_results)
```

## Benchmarks and Results

| Benchmark | Task | Samples | Privacy Protection | Task Performance | Command |
|-----------|------|---------|-------------------|------------------|---------|
| **Vendor-Neutral Synthetic** | Interest evaluation on behavioral data (medical, financial, education) | 300 | 100.0% (300/300) | 85.0% accuracy (255/300) | `python3 benchmarks/neutral_benchmark.py --benchmark all --domains all --num-samples 100` |
| **Text Masking** ([ai4privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)) | PII masking across 54 entity types | 1,000 | 28.8% full protection (288/1000)<br>71.2% partial leakage | 4,012 PII entities tested | `python3 benchmarks/public_datasets_simple.py --num-samples 1000` |
| **Question Answering** ([PUPA](https://github.com/Columbia-NLP-Lab/PAPILLON)) | Response generation without PII leakage | 901 | 81.2% protected (3904/4806 PII units)<br>18.8% leakage | 100.0% response success<br>(vs. PAPILLON 85.5%) | `python3 benchmarks/pupa_benchmark.py --simulate --num-samples 901` |
| **Document Sanitization** ([TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark)) | Court document anonymization | 1,268 | 99.9% direct IDs (1267/1268)<br>99.9% quasi IDs (3806/3807)<br>83.7% overall (5308/6340) | PERSON/ORG/LOC: 99.9%<br>CODE: 18.9% | `python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 1268` |
| **Differential Privacy Comparison** | Adversarial privacy testing (canary exposure, MIA) | 100 | 0.0% canary exposure (0/10)<br>98.5% MIA resistance | Comparable to DP (ε=1.0) | `python3 benchmarks/dp_benchmark.py --num-samples 100` |

## Comparison with Differential Privacy

| Aspect | This Approach | Differential Privacy |
|--------|--------------|----------------------|
| **Privacy Guarantee** | Empirical (benchmark-measured) | Formal (mathematical proof) |
| **Training Required** | No | Yes (DP-SGD) |
| **Utility** | Measured on benchmarks | Typically experiences utility degradation |
| **Applicability** | Inference-time only | Training and/or inference |

**Note:** Differential Privacy provides formal mathematical guarantees (ε, δ), while this approach provides empirical privacy measurements on specific benchmarks. Both approaches have complementary strengths depending on use case requirements.

## References

### Datasets

1. **ai4privacy/pii-masking-200k**
   - Source: [HuggingFace Dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
   - Description: 200K+ samples with 54 PII types for privacy research

2. **PUPA (Private User Prompt Annotations)**
   - Source: [GitHub Repository](https://github.com/Columbia-NLP-Lab/PAPILLON)
   - Description: 901 real-world user-agent interactions from WildChat corpus

3. **TAB (Text Anonymization Benchmark)**
   - Source: [GitHub Repository](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
   - Description: 1,268 ECHR court cases with manual PII annotations

### Papers

1. **PAPILLON**
   - Li et al. (2025). "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles." *NAACL 2025*.

2. **TAB**
   - Pilán et al. (2022). "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization." *ACL 2022 Findings*.

3. **SanText**
   - Yue et al. (2021). "Differential Privacy for Text Analytics via Natural Text Sanitization." *ACL 2021 Findings*.
   - [GitHub Repository](https://github.com/xiangyue9607/SanText)

### Models

The pipeline is designed to work with any LLM API. Models used in benchmark evaluations:
- gpt-oss-120b
- DeepSeek-V3.1
- Qwen3-32B
- DeepSeek-V3-0324

### Documentation

- **[Pipeline Architecture](docs/ENSEMBLE_PIPELINE_EXPLAINED.md)** - Complete technical explanation of the ensemble-redaction approach
- **[Benchmark Details](docs/BENCHMARKS.md)** - Detailed benchmark guide with usage instructions
