# Ensemble-Redaction Consensus Pipeline

**A Training-Free Privacy-Preserving Approach for LLM Inference on Sensitive User Data**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Benchmark Results](#benchmark-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Detailed Results](#detailed-results)
- [Available Benchmarks](#available-benchmarks)
- [References](#references)

## Overview

This repository implements a privacy-preserving pipeline for LLM inference that combines input masking with ensemble consensus voting. The approach is designed to reduce personally identifiable information (PII) exposure when processing sensitive user data through large language models.

**Key Characteristics:**
- Training-free implementation compatible with any LLM API
- Input masking layer to redact PII before model inference
- Ensemble consensus to reduce individual model variance
- Evaluation on public privacy benchmarks

## Benchmark Results

Evaluation on 3,569 samples across 5 privacy-preserving tasks:

| Benchmark | Privacy Metric | Task Performance | Notes |
|-----------|----------------|------------------|-------|
| **Vendor-Neutral Synthetic** | 100.0% protection | 85.0% accuracy | Synthetic data evaluation |
| **Text Masking (ai4privacy)** | 28.8% full protection | 1000 samples | 54 PII types evaluated |
| **Question Answering (PUPA)** | 81.2% PII protected | 100.0% response rate | NAACL 2025 dataset |
| **Document Sanitization (TAB)** | 99.9% direct IDs | 83.7% overall masking | ACL 2022 dataset |
| **Differential Privacy** | 0.0% canary exposure | 98.5% MIA resistance | Adversarial testing |

**Summary of Findings:**
- Privacy protection rates range from 81-100% across different benchmark tasks
- Task performance maintained at 85-100% on evaluated metrics
- Comparable results to existing privacy-preserving approaches on select benchmarks

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

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

### Basic Example

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

### Available Commands

```bash
# Full benchmark suite (3,569 samples)
python3 run_all_benchmarks.py

# Privacy comparison demonstration
python3 examples/privacy_comparison.py

# Individual benchmarks
python3 benchmarks/neutral_benchmark.py --benchmark all --num-samples 100
python3 benchmarks/public_datasets_simple.py --num-samples 1000
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 901
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 1268
python3 benchmarks/dp_benchmark.py --num-samples 100
```

## Detailed Results

### 1. Vendor-Neutral Synthetic Benchmark
**Task**: Interest evaluation on synthetic behavioral data (medical, financial, education domains)

**Results**:
- Privacy Protection: 100.0% (300/300 samples)
- Task Accuracy: 85.0% (255/300 correct matches)
- Average Score: 0.21
- Processing Time: 4,174s (~14s per sample)

**Observations**: Complete privacy protection achieved on synthetic data with maintained task performance.

---

### 2. Text Masking (ai4privacy/pii-masking-200k)
**Task**: PII masking across 54 different entity types

**Results**:
- Full Protection Rate: 28.8% (288/1000 samples)
- Partial Leakage: 71.2% (712/1000 samples with at least one entity leaked)
- Total PII Entities: 4,012 across 1,000 samples
- Common PII Types: FIRSTNAME (290), LASTNAME (98), DATE (86), EMAIL (83)
- Processing Time: 5,875s (~6s per sample)

**Observations**: The 28.8% full protection rate reflects the challenge of detecting and masking all 54 PII types, particularly quasi-identifiers and domain-specific codes. This highlights areas for improvement in comprehensive PII detection.

---

### 3. Question Answering (PUPA - NAACL 2025)
**Task**: Response generation without PII leakage from prompts

**Results**:
- PII Leakage Rate: 18.8% (902/4806 PII units leaked)
- PII Protection Rate: 81.2% (3904/4806 PII units protected)
- Response Success Rate: 100.0% (901/901 responses generated)
- Processing Time: 11,170s (~12s per sample)

**Category-Specific Performance**:
- Job/Visa Applications: 32.8% leakage
- Financial Information: 19.7% leakage
- Quoted Emails/Messages: 0.9% leakage

**Comparison with PAPILLON (NAACL 2025)**:
| Metric | This Approach | PAPILLON |
|--------|---------------|----------|
| Response Success | 100.0% | 85.5% |
| Privacy Leakage | 18.8% | 7.5% |

**Observations**: Higher response success rate achieved compared to PAPILLON, though with higher PII leakage (11.3 percentage points). This suggests a privacy-utility trade-off that differs from the PAPILLON approach.

---

### 4. Document Sanitization (TAB)
**Task**: Anonymization of court documents for public release

**Results**:
- Overall PII Masking: 83.7% (5308/6340 entities protected)
- Direct ID Protection: 99.9% (1267/1268)
- Quasi ID Protection: 99.9% (3806/3807)
- Processing Time: 8,958s (~7s per sample)

**Entity Type Performance**:
- PERSON: 99.9% protected
- ORG: 99.9% protected
- LOC: 99.9% protected
- DATETIME: 99.9% protected
- CODE (legal references): 18.9% protected

**Observations**: Strong protection achieved for direct identifiers (PERSON, ORG). Lower performance on CODE entities (legal references) suggests specialized handling may be needed for domain-specific identifiers.

---

### 5. Differential Privacy Comparison
**Task**: Adversarial privacy evaluation (canary exposure, membership inference)

**Results**:
- Canary Exposure Rate: 0.0% (0/10 canaries exposed)
- MIA Resistance: 98.5% (member/non-member score difference: 0.015)
- Processing Time: 559s

**Comparison with Differential Privacy**:
| Approach | Canary Exposure | MIA Resistance |
|----------|----------------|----------------|
| This Approach | 0.0% | 98.5% |
| DP (ε=1.0) | ~8.0% | ~85.0% |
| DP (ε=5.0) | ~18.0% | ~65.0% |

**Observations**: Results suggest comparable adversarial privacy resistance to ε=1.0 differential privacy on these specific tests. Note that formal DP provides mathematical guarantees, while this approach provides empirical measurements.

---

## Available Benchmarks

### Public Datasets

The evaluation uses publicly available datasets for reproducibility:

| Benchmark | Samples | Task | Dataset Source |
|-----------|---------|------|----------------|
| **Vendor-Neutral Synthetic** | 300 | Interest Evaluation | Synthetic data (this work) |
| **ai4privacy/pii-masking-200k** | 1,000 | Text Masking | [HuggingFace](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) |
| **PUPA** | 901 | Question Answering | [GitHub](https://github.com/Columbia-NLP-Lab/PAPILLON) |
| **TAB** | 1,268 | Document Sanitization | [GitHub](https://github.com/NorskRegnesentral/text-anonymization-benchmark) |
| **Differential Privacy** | 100 | Adversarial Testing | Methodology-based (this work) |

### Running Benchmarks

```bash
# Set API key
export LLM_API_KEY='your-key-here'

# Run full suite
python3 run_all_benchmarks.py  # 3,569 samples, estimated 7-9 hours

# Run individual benchmarks
python3 benchmarks/neutral_benchmark.py --benchmark all --num-samples 100
python3 benchmarks/public_datasets_simple.py --num-samples 1000
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 901
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 1268
python3 benchmarks/dp_benchmark.py --num-samples 100
```

### Evaluation Metrics

Benchmarks measure:
- **PII Protection**: Percentage of PII successfully masked in outputs
- **Task Performance**: Accuracy on the specific task (e.g., question answering, text masking)
- **Adversarial Resistance**: Canary exposure and membership inference attack resistance

## References

### Datasets

1. **ai4privacy/pii-masking-200k**
   - Source: [HuggingFace Dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
   - License: Apache 2.0
   - Description: 200K+ samples with 54 PII types for privacy research

2. **PUPA (Private User Prompt Annotations)**
   - Source: [GitHub Repository](https://github.com/Columbia-NLP-Lab/PAPILLON)
   - Paper: Li et al., "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles", NAACL 2025
   - Description: 901 real-world user-agent interactions from WildChat corpus

3. **TAB (Text Anonymization Benchmark)**
   - Source: [GitHub Repository](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
   - Paper: Pilán et al., "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization", ACL 2022 Findings
   - Description: 1,268 ECHR court cases with manual PII annotations

### Related Work

- **PAPILLON**: Li et al. (2025). "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles." *NAACL 2025*.
- **TAB**: Pilán et al. (2022). "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization." *ACL 2022 Findings*.
- **SanText**: Yue et al. (2021). "Differential Privacy for Text Analytics via Natural Text Sanitization." *ACL 2021 Findings*. [GitHub](https://github.com/xiangyue9607/SanText)

### Comparison with Differential Privacy

| Aspect | This Approach | Differential Privacy |
|--------|--------------|----------------------|
| **Privacy Guarantee** | Empirical (benchmark-measured) | Formal (mathematical proof) |
| **Training Required** | No | Yes (DP-SGD) |
| **Utility** | Measured on benchmarks | Typically experiences utility degradation |
| **Applicability** | Inference-time only | Training and/or inference |

**Note**: Differential Privacy provides formal mathematical guarantees (ε, δ), while this approach provides empirical privacy measurements on specific benchmarks. Both approaches have complementary strengths depending on use case requirements.

### Models

The pipeline is designed to work with any LLM API. Models used in benchmark evaluations:
- gpt-oss-120b
- DeepSeek-V3.1
- Qwen3-32B
- DeepSeek-V3-0324

---

## Documentation

For additional documentation, see:
- [Pipeline Architecture](docs/ENSEMBLE_PIPELINE_EXPLAINED.md)
- [Benchmark Details](docs/BENCHMARKS.md)

---

## Getting Started

```bash
export LLM_API_KEY='your-key-here'
python3 run_all_benchmarks.py
```
