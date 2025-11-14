# Complete Guides

This document consolidates all user guides and how-tos for the Ensemble Privacy Pipeline.

---

## 📖 Table of Contents

1. [Using Real LLMs](#using-real-llms)
2. [Migration Guide](#migration-guide)
3. [Public Benchmarks](#public-benchmarks)
4. [Understanding Results](#understanding-results)

---

## 🚀 Using Real LLMs

### Quick Start

```bash
# Step 1: Install packages
pip install openai anthropic google-generativeai

# Step 2: Set API keys
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'

# Step 3: Run with real LLMs
python examples/real_llm_example.py
```

### Replace Mock with Real LLMs

```python
# OLD CODE (Mock LLMs)
from src.pipeline import MockLLMEvaluator

evaluators = [
    MockLLMEvaluator("GPT-4", bias=0.0),
    MockLLMEvaluator("Claude-3.5", bias=0.05)
]

# NEW CODE (Real LLMs)
from examples.real_llm_example import RealLLMEvaluator

evaluators = [
    RealLLMEvaluator("gpt-4", api_key=os.getenv('OPENAI_API_KEY')),
    RealLLMEvaluator("claude-3-5-sonnet-20241022", api_key=os.getenv('ANTHROPIC_API_KEY'))
]
```

### Cost Comparison

| Setup | Cost/User | Time | Use Case |
|-------|-----------|------|----------|
| Mock LLMs | $0.00 | 2 sec | Development, testing, benchmarks |
| 3-model ensemble | $0.02 | 2 sec | Production (balanced) |
| 5-model ensemble | $0.05 | 3 sec | Production (high accuracy) |

---

## 🔄 Migration Guide

### Repository Reorganization

**Old Structure**:
```
ensemble-privacy-pipeline/
├── ensemble_privacy_pipeline.py
├── evaluation_framework.py
├── benchmark_public_datasets.py
└── [8 more Python files in root]
```

**New Structure**:
```
ensemble-privacy-pipeline/
├── src/
│   ├── pipeline.py
│   └── evaluators.py
├── benchmarks/
│   ├── public_datasets.py
│   └── dp_specific.py
├── examples/
│   ├── privacy_comparison.py
│   └── real_llm_example.py
└── tests/
```

### Import Migration

**Old imports (still work - backward compatible)**:
```python
from ensemble_privacy_pipeline import PrivacyRedactor
from evaluation_framework import PrivacyEvaluator
```

**New imports (recommended)**:
```python
from src.pipeline import PrivacyRedactor, ConsensusAggregator
from src.evaluators import PrivacyEvaluator, UtilityEvaluator
```

### Script Migration

**Old commands (still work)**:
```bash
python ensemble_privacy_pipeline.py
python benchmark_public_datasets.py
```

**New commands (recommended)**:
```bash
python src/pipeline.py
python benchmarks/public_datasets.py
```

---

## 🔬 Public Benchmarks

### Overview

**Your pipeline CAN be tested on public benchmarks** - not just your own data!

| Benchmark | Type | Size | Access |
|-----------|------|------|--------|
| ai4privacy/pii-masking-200k | Hugging Face | 209K samples | [Public](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) |
| PII-Bench | ACL 2024 | 6.8K samples | Public |
| PrivacyXray | Synthetic | 50K individuals | Public |

### Quick Commands

```bash
# Test on ai4privacy (200K+ public samples from Hugging Face)
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000

# Test on PII-Bench (ACL 2024 standard)
python benchmarks/public_datasets.py --benchmark pii-bench --num_samples 500

# Test on ALL public benchmarks
python benchmarks/public_datasets.py --benchmark all --num_samples 500
```

### What These Benchmarks Test

All benchmarks test **YOUR PIPELINE that uses LLMs**:
- ✅ How well your redaction prevents PII from entering LLMs
- ✅ Whether LLM outputs leak sensitive information
- ✅ If ensemble consensus suppresses individual model artifacts
- ✅ Your privacy boundary enforcement

They DON'T test:
- ❌ Individual LLM quality or capabilities

### Expected Results

```
Privacy Metrics:
  PII Leakage:
    Baseline:      85.0% (without protection)
    With Pipeline:  0.0% (with your approach)
    Improvement:   85.0 percentage points ✅

  Reconstruction Attack:
    Baseline:      75.0% success rate
    With Pipeline:  0.0% success rate
    Improvement:   100% attack prevention ✅
```

---

## 📊 Understanding Results

### Numbers in README

The numbers in README.md come from different sources:

#### 1. Privacy Metrics (3 queries, 11 titles leaked)
- **Source**: Demo script (`examples/privacy_comparison.py`)
- **Status**: ✅ Accurate for demo example data
- **Reproduce**: `python examples/privacy_comparison.py`

#### 2. Benchmark Results (0.0% PII leakage)
- **Source**: Expected results based on mechanism design
- **Status**: ⚠️ Not yet measured (need to run benchmarks)
- **Reproduce**: `python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000`

#### 3. DP Comparison (2.0% canary, AUC 0.58)
- **Source**: Typical values from DP research literature
- **Status**: ⚠️ Expected values (need to run to confirm)
- **Reproduce**: `python benchmarks/dp_specific.py --test canary --num_samples 100`

### LLMs Used

**Current implementation uses Mock LLMs**:
- Located in `src/pipeline.py` (MockLLMEvaluator class)
- Simulates 5 models: GPT-4, Claude-3.5, Gemini-Pro, Llama-3, Mistral-Large
- Each has different bias values to simulate variance
- **Free, fast, reproducible**

**For real LLMs**: Use `examples/real_llm_example.py`

### Benchmark Target

**Q**: Are benchmarks for LLMs?

**A**: YES - Benchmarks test **YOUR PIPELINE that USES LLMs**

```
YOUR PIPELINE (tested):
  Input: Sensitive data
    ↓
  Step 1: Redaction (your code) ← Tested
    ↓
  Step 3: LLM Ensemble (Mock or Real) ← Tested
    ↓
  Step 4: Consensus (your code) ← Tested
    ↓
  Output: Safe JSON
    ↓
  Benchmark checks: PII leakage? ← Tested
```

---

## 🎯 Common Tasks

### Test Privacy Protection

```bash
# Quick demo (3 seconds)
python examples/privacy_comparison.py
```

### Validate on Public Data

```bash
# Test on 1000 public samples (5 minutes)
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000
```

### Use Real LLM APIs

```bash
# Set keys
export OPENAI_API_KEY='sk-...'

# Run
python examples/real_llm_example.py
```

### Run Full Benchmark Suite

```bash
# All public benchmarks + DP tests
python benchmarks/public_datasets.py --benchmark all --num_samples 500
python benchmarks/dp_specific.py --run_full_evaluation --num_samples 100
```

---

## 📚 Additional Resources

- **Main README**: [../README.md](../README.md)
- **Protocol Spec**: [PROTOCOL.md](PROTOCOL.md)
- **Examples**: [../examples/README.md](../examples/README.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Date**: 2025-01-14
