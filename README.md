# Ensemble-Redaction Consensus Pipeline

**Training-Free Privacy-Preserving LLM Pipeline for Sensitive User Data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Quick Results](#quick-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Comparison with Differential Privacy](#comparison-with-differential-privacy)
- [References](#references)
- [License](#license)

## Overview

Privacy-preserving LLM pipeline using input masking and ensemble consensus voting. The pipeline ensures that LLMs never process raw personally identifiable information (PII).

**Key Features:**
- Training-free: Works with any LLM API without model training
- Zero PII leakage: Input masking prevents exposure of sensitive data
- Utility evaluation: Benchmarks measure task performance on protected pipeline
- Production-ready: Comprehensive benchmark suite available

## Quick Results

The pipeline is designed to:
- **Prevent PII leakage**: Input masking ensures LLMs never process raw sensitive data
- **Block reconstruction attacks**: Ensemble consensus filters out identifiable information
- **Preserve utility**: Benchmarks evaluate task performance to measure utility preservation

Run the benchmarks to see actual results on your data and models.

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

Run the demo pipeline:

```bash
python3 run_demo_pipeline.py
```

Run the full benchmark suite:

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

# 2. Evaluate with ensemble (4 diverse models)
api_key = os.getenv("LLM_API_KEY")
ensemble_models = ["model-1", "model-2", "model-3", "model-4"]  # Use 4 diverse LLMs
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
# Full benchmark suite (3,569 samples, 5 benchmarks)
python3 run_all_benchmarks.py

# Privacy comparison demo
python3 examples/privacy_comparison.py

# Individual benchmarks
python3 benchmarks/neutral_benchmark.py --benchmark all --num-samples 100
python3 benchmarks/public_datasets_simple.py --num-samples 1000
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 901
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 1268
python3 benchmarks/dp_benchmark.py --num-samples 100

# Legacy demos
python3 run_benchmarks.py --benchmark all --num-samples 20
python3 run_demo_pipeline.py
```

## Benchmarks

### Full Benchmark Suite

Run all benchmarks with a single command:

```bash
export LLM_API_KEY='your-key-here'
python3 run_all_benchmarks.py  # 3,569 samples, 7-9 hours
```

### Included Benchmarks

All benchmarks use public datasets and test different privacy-preserving tasks:

| Benchmark | Samples | Task | Description |
|-----------|---------|------|-------------|
| **Vendor-Neutral Synthetic** | 300 | Interest Evaluation | Medical, financial, education behavioral data |
| **[ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)** | 1,000 | Text Masking | 54 PII types - test masking quality |
| **[PUPA (NAACL 2025)](https://github.com/Columbia-NLP-Lab/PAPILLON)** | 901 | Question Answering | Answer questions without leaking PII |
| **[TAB - Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark)** | 1,268 | Document Sanitization | Anonymize court documents for publication |
| **Differential Privacy** | 100 | Interest Evaluation | Canary exposure, MIA, DP comparison |

### Dataset References

1. **Vendor-Neutral Synthetic Benchmark**
   - *Synthetic benchmark generated for this project*
   - No external dataset required

2. **[ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)**
   - **Source**: [HuggingFace Dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
   - **License**: Apache 2.0
   - **Description**: 200K+ samples with 54 PII types, designed for PII research

3. **[PUPA (NAACL 2025)](https://github.com/Columbia-NLP-Lab/PAPILLON)**
   - **Source**: [GitHub Repository](https://github.com/Columbia-NLP-Lab/PAPILLON)
   - **Paper**: "PAPILLON: Privacy Preservation from Internet-based and Local Language Model Ensembles" (Li et al., NAACL 2025)
   - **Description**: 901 real-world user-agent interactions from WildChat corpus

4. **[TAB - Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark)**
   - **Source**: [GitHub Repository](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
   - **Paper**: Pilán et al. (2022) "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization"
   - **Description**: 1,268 English-language court cases from European Court of Human Rights with manual PII annotations

5. **Differential Privacy Benchmark**
   - *Methodology-based benchmark*
   - Tests canary exposure (PrivLM-Bench style) and Membership Inference Attacks (MIA)
   - Compares against formal DP (ε=1.0, ε=5.0)

### Benchmark Metrics

The benchmarks evaluate:
- **PII Protection**: Percentage of PII successfully masked in outputs
- **Utility Accuracy**: Task performance on protected pipeline
- **Canary Exposure**: Ability to extract unique identifiers
- **MIA Resistance**: Resistance to Membership Inference Attacks

Run the benchmarks to get actual results for your specific setup.

## Comparison with Differential Privacy

| Aspect | This Approach | Differential Privacy |
|--------|--------------|----------------------|
| **Privacy Guarantee** | Empirical (benchmark-validated) | Formal (mathematical) |
| **Training Required** | No | Yes (DP-SGD) |
| **Utility** | High (measured on benchmarks) | Typically 20-50% utility loss |
| **Use Case** | Production APIs | Research, formal guarantees |

## References

### Datasets

- **ai4privacy/pii-masking-200k**: [HuggingFace Dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) - 200K+ samples with 54 PII types, Apache 2.0 License
- **PUPA (Private User Prompt Annotations)**: [GitHub Repository](https://github.com/Columbia-NLP-Lab/PAPILLON) - 901 real-world user-agent interactions from WildChat corpus
- **TAB (Text Anonymization Benchmark)**: [GitHub Repository](https://github.com/NorskRegnesentral/text-anonymization-benchmark) - 1,268 ECHR court cases with manual PII annotations

### Papers

- **PAPILLON**: Li et al. (2025). "PAPILLON: Privacy Preservation from Internet-based and Local Language Model Ensembles." *NAACL 2025*.
- **TAB**: Pilán et al. (2022). "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization."
- **SanText**: Yue et al. (2021). "Differential Privacy for Text Analytics via Natural Text Sanitization." *ACL 2021 Findings*. [GitHub](https://github.com/xiangyue9607/SanText)

### Models

The pipeline is designed to work with any LLM API. Example models used in benchmarks:
- gpt-oss-120b
- DeepSeek-V3.1
- Qwen3-32B
- DeepSeek-V3-0324

### Methodologies

- **PrivLM-Bench**: Canary exposure testing methodology for privacy evaluation
- **WildChat**: Source corpus for PUPA dataset (user-agent interaction data)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Getting Started

```bash
export LLM_API_KEY='your-key-here'
python3 run_all_benchmarks.py
```
