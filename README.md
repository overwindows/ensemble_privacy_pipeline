# Ensemble-Redaction Consensus Pipeline

**Training-Free Privacy-Preserving LLM Pipeline for Sensitive User Data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Privacy-preserving LLM pipeline using **input masking + ensemble consensus voting**. LLMs never see raw PII.

## 📊 Quick Results

| Metric | Without | With Pipeline |
|--------|---------|---------------|
| PII Leakage | 75-100% | 0% |
| Reconstruction Attack | Success | Failed |
| Utility (Accuracy) | 0.85 | 0.85 (same) |
| Cost | - | $0.05/user |

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/overwindows/ensemble-privacy-pipeline.git
cd ensemble-privacy-pipeline
pip3 install -r requirements.txt

# Set API key (example uses any OpenAI-compatible API)
export LLM_API_KEY='your-key-here'

# Run demo
python3 run_demo_pipeline.py
```

## 📋 Available Commands

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

## 📖 Basic Usage

```python
from src.privacy_core import PrivacyRedactor, ConsensusAggregator
from examples.real_llm_example import RealLLMEvaluator

# 1. Redact sensitive data
redactor = PrivacyRedactor()
masked_data = redactor.redact_user_data(raw_user_data)

# 2. Evaluate with ensemble (4 diverse models)
api_key = os.getenv("LLM_API_KEY")
ensemble_models = ["model-1", "model-2", "model-3", "model-4"]  # Use 4 diverse LLMs
all_results = [RealLLMEvaluator(m, api_key).evaluate_interest(masked_data, topics) for m in ensemble_models]

# 3. Aggregate with consensus
aggregator = ConsensusAggregator()
final_output = aggregator.aggregate_median(all_results)
```

## 🧪 Benchmarks

### Full Benchmark Suite

```bash
export LLM_API_KEY='your-key-here'
python3 run_all_benchmarks.py  # 3,569 samples, $300-380, 7-9 hours
```

**Included Benchmarks** (all public datasets):
| Benchmark | Samples | Type |
|-----------|---------|------|
| Vendor-Neutral Synthetic | 300 | Synthetic (medical, financial, education) |
| ai4privacy/pii-masking-200k | 1,000 | Public (54 PII types) |
| PUPA (NAACL 2025) | 901 | Simulated (WildChat-style) |
| TAB - Text Anonymization | 1,268 | Simulated (ECHR court cases) |
| Differential Privacy | 100 | Synthetic (Canary, MIA, DP) |

### Expected Results
- PII Protection: >95%
- Utility Accuracy: ~85%
- Canary Exposure: <5%
- MIA AUC: 0.58 (close to random)

## 🆚 vs. Differential Privacy (DP)

| Aspect | This Approach | Differential Privacy |
|--------|--------------|----------------------|
| Privacy Guarantee | Empirical (benchmark-validated) | Formal (mathematical) |
| Training Required | ❌ No | ✅ Yes (DP-SGD) |
| Utility Loss | 0% | 20-50% |
| Cost | $0.05/user | $$$ training |
| Use Case | Production APIs | Research, formal guarantees |

## 📂 Repository Structure

```
ensemble-privacy-pipeline/
├── src/                                # Core privacy components
├── benchmarks/                         # Benchmark scripts
├── examples/                           # Usage examples
├── docs/                               # Documentation
└── run_all_benchmarks.py              # Main evaluation script
```

## 📚 Documentation

- **[Documentation Index](docs/README.md)** - All documentation links
- **[How It Works](docs/ENSEMBLE_PIPELINE_EXPLAINED.md)** - Complete pipeline explanation
- **[Benchmark Guide](docs/BENCHMARK_REVIEW.md)** - Comprehensive evaluation suite
- **[Benchmarks Usage](benchmarks/README.md)** - Individual benchmark scripts
- **[Contributing](docs/CONTRIBUTING.md)** - Contribution guidelines

## 📄 License

MIT License

---

**Ready to start?**

```bash
export LLM_API_KEY='your-key-here'
python3 run_all_benchmarks.py
```
