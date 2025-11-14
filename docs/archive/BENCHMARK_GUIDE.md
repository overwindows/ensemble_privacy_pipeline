# Public Benchmark Evaluation Guide

This guide explains how to evaluate the Ensemble-Redaction Consensus Pipeline on industry-standard public privacy benchmarks.

---

## 📊 Supported Benchmarks

### 1. **ai4privacy/pii-masking-200k** (Recommended)
- **Source**: Hugging Face
- **Size**: 209,261 rows (200K+ samples)
- **Languages**: English, French, German, Italian
- **PII Classes**: 54 different types of sensitive data
- **License**: Apache 2.0
- **Use Case**: Real-world PII detection and masking

**Citation**:
```bibtex
@dataset{ai4privacy_pii_masking_200k,
  title={PII Masking 200K Dataset},
  author={AI4Privacy},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/ai4privacy/pii-masking-200k}
}
```

### 2. **PII-Bench** (2025)
- **Source**: arXiv 2025 (synthetic data following paper methodology)
- **Size**: Configurable (default 500 samples)
- **PII Types**: 55 categories (medical, financial, legal, personal)
- **Focus**: Query-aware privacy protection
- **Reference**: [PII-Bench Paper](https://arxiv.org/html/2502.18545v1)

**Citation**:
```bibtex
@article{pii_bench_2025,
  title={PII-Bench: Evaluating Query-Aware Privacy Protection Systems},
  year={2025},
  journal={arXiv preprint arXiv:2502.18545}
}
```

### 3. **PrivacyXray** (2025)
- **Source**: Synthetic data following PrivacyXray methodology
- **Size**: Configurable (default 500 synthetic individuals)
- **PII Types**: 16 comprehensive categories
- **Coverage**: 400,000 unique PII entries (in full dataset)
- **Focus**: Comprehensive PII profiling

**Citation**:
```bibtex
@article{privacyxray_2025,
  title={PrivacyXray: Detecting Privacy Breaches in LLMs},
  year={2025},
  journal={arXiv preprint}
}
```

---

## 🚀 Quick Start

### Installation

```bash
# Install core dependencies
pip install numpy scikit-learn

# Install benchmark dataset loaders
pip install datasets huggingface_hub

# Optional: Install for real LLM evaluation
pip install openai anthropic google-generativeai
```

### Run Single Benchmark

```bash
# Evaluate on ai4privacy dataset (1000 samples)
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 1000

# Evaluate on PII-Bench (500 samples)
python benchmark_public_datasets.py --benchmark pii-bench --num_samples 500

# Evaluate on PrivacyXray (500 samples)
python benchmark_public_datasets.py --benchmark privacyxray --num_samples 500
```

### Run All Benchmarks

```bash
# Evaluate on all benchmarks
python benchmark_public_datasets.py --benchmark all --num_samples 1000
```

### Custom Output File

```bash
# Save results to custom file
python benchmark_public_datasets.py \
  --benchmark ai4privacy \
  --num_samples 2000 \
  --output my_results.json
```

---

## 📁 Output Format

Results are saved in JSON format with the following structure:

```json
{
  "config": {
    "benchmark": "ai4privacy",
    "num_samples": 1000,
    "timestamp": "2025-01-14 10:30:00"
  },
  "results": {
    "ai4privacy": {
      "benchmark": "ai4privacy",
      "num_samples": 1000,
      "privacy_metrics": {
        "pii_leakage": {
          "baseline": 0.85,
          "with_privacy": 0.0,
          "improvement": 0.85,
          "improvement_pct": 85.0
        },
        "reconstruction_attack": {
          "baseline": 0.75,
          "with_privacy": 0.0,
          "improvement": 0.75,
          "improvement_pct": 75.0
        }
      },
      "detailed": {
        "baseline_pii": {...},
        "privacy_pii": {...}
      }
    }
  }
}
```

---

## 📊 Interpreting Results

### Privacy Metrics

#### **PII Leakage Rate**
- **Definition**: Percentage of outputs containing personally identifiable information
- **Target**: 0% (no PII should leak)
- **Calculation**: `(outputs_with_pii / total_outputs) * 100`

**Example**:
```
Baseline (no privacy):     85.0%  ❌ Most outputs leak PII
With Privacy Pipeline:      0.0%  ✅ No PII leaked
Improvement:              85.0%  ✅ Excellent
```

#### **Reconstruction Attack Success Rate**
- **Definition**: Percentage of cases where attacker can reconstruct original queries from outputs
- **Target**: 0% (reconstruction should be impossible)
- **Calculation**: Measures term overlap between outputs and ground truth queries

**Example**:
```
Baseline Success:         75.0%  ❌ Attacker can reconstruct queries
With Privacy:              0.0%  ✅ Reconstruction failed
Improvement:              75.0%  ✅ Excellent
```

### Success Criteria

| Improvement | Rating | Verdict |
|-------------|--------|---------|
| ≥80% | ✅ Excellent | Pipeline provides strong privacy protection |
| 60-80% | ✅ Good | Pipeline provides good privacy protection |
| 40-60% | ⚠️ Moderate | Room for improvement |
| <40% | ❌ Weak | Pipeline needs significant improvement |

---

## 🧪 Detailed Benchmark Comparison

### Benchmark Characteristics

| Benchmark | Samples | PII Types | Language | Source | Best For |
|-----------|---------|-----------|----------|--------|----------|
| **ai4privacy** | 200K+ | 54 | Multi | Real-world | Production validation |
| **PII-Bench** | Custom | 55 | English | Synthetic | Query-aware evaluation |
| **PrivacyXray** | Custom | 16 | English | Synthetic | Comprehensive profiling |

### Recommended Evaluation Strategy

1. **Start with ai4privacy** (1000 samples)
   - Most realistic real-world data
   - Industry-standard benchmark
   - Multi-language support

2. **Validate with PII-Bench** (500 samples)
   - Tests query-aware privacy
   - Covers diverse PII categories
   - Research-grade evaluation

3. **Deep-dive with PrivacyXray** (500 samples)
   - Comprehensive PII coverage
   - Tests against full profiles
   - Advanced attack scenarios

---

## 🔬 Advanced Usage

### Programmatic Evaluation

```python
from benchmark_public_datasets import PublicBenchmarkEvaluator

# Initialize evaluator
evaluator = PublicBenchmarkEvaluator()

# Run evaluation
results = evaluator.evaluate_on_benchmark(
    benchmark_name='ai4privacy',
    num_samples=1000
)

# Access metrics
pii_improvement = results['privacy_metrics']['pii_leakage']['improvement_pct']
print(f"PII Leakage Improvement: {pii_improvement:.1f}%")
```

### Custom Dataset Integration

```python
from benchmark_public_datasets import PublicBenchmarkLoader

loader = PublicBenchmarkLoader()

# Load ai4privacy dataset
dataset = loader.load_ai4privacy_dataset(
    num_samples=5000,
    language='en'  # 'en', 'de', 'fr', 'it', or 'all'
)

# Process samples
for sample in dataset:
    print(f"Sample ID: {sample['id']}")
    print(f"PII Types: {sample['pii_types']}")
    print(f"Queries: {sample['queries']}")
```

---

## 📈 Expected Results

Based on protocol evaluation, you should see:

### ai4privacy/pii-masking-200k

```
PII Leakage:
  Baseline:           85-100%
  With Privacy:       0-5%
  Improvement:        ≥80%

Reconstruction:
  Baseline:           70-90%
  With Privacy:       0-5%
  Improvement:        ≥70%
```

### PII-Bench

```
PII Leakage:
  Baseline:           80-95%
  With Privacy:       0-5%
  Improvement:        ≥75%

Reconstruction:
  Baseline:           65-85%
  With Privacy:       0-5%
  Improvement:        ≥60%
```

### PrivacyXray

```
PII Leakage:
  Baseline:           90-100%
  With Privacy:       0-5%
  Improvement:        ≥85%

Reconstruction:
  Baseline:           75-95%
  With Privacy:       0-5%
  Improvement:        ≥70%
```

---

## 🐛 Troubleshooting

### Issue: Cannot load ai4privacy dataset

**Error**: `ModuleNotFoundError: No module named 'datasets'`

**Solution**:
```bash
pip install datasets huggingface_hub
```

### Issue: Hugging Face authentication required

**Error**: `Repository requires authentication`

**Solution**:
```bash
# Login to Hugging Face (one-time)
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN="your_token_here"
```

### Issue: Out of memory when loading large dataset

**Solution**:
```bash
# Reduce number of samples
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 500

# Or use streaming mode (modify code)
dataset = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)
```

### Issue: Slow evaluation

**Tip**: Start with smaller sample sizes
```bash
# Quick test
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 100

# Full evaluation (slower)
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 10000
```

---

## 📚 References

1. **ai4privacy PII-Masking-200k**
   - Dataset: https://huggingface.co/datasets/ai4privacy/pii-masking-200k
   - License: Apache 2.0

2. **PII-Bench (2025)**
   - Paper: https://arxiv.org/html/2502.18545v1
   - Focus: Query-aware privacy protection

3. **PrivacyXray (2025)**
   - Paper: https://arxiv.org/html/2506.19563v1
   - Focus: Semantic consistency and probability certainty

4. **PII-Scope (2024)**
   - Paper: https://arxiv.org/html/2410.06704v1
   - Focus: Training data PII leakage

---

## 🤝 Contributing

To add a new benchmark:

1. Add loader method to `PublicBenchmarkLoader` class
2. Implement data conversion to standard format
3. Add benchmark option to CLI arguments
4. Update this README with benchmark details
5. Submit pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This evaluation framework is released under MIT License.

Individual benchmark datasets have their own licenses:
- ai4privacy: Apache 2.0
- PII-Bench: Research use (cite paper)
- PrivacyXray: Research use (cite paper)

---

**Built with privacy in mind. Evaluate with confidence.** 🛡️
