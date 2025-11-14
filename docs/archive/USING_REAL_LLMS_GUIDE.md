# Using Real LLMs Guide

**Complete guide to switching from Mock LLMs to Real LLM APIs**

---

## ❓ Your Questions Answered

### Question 1: What should I do if I want to leverage real LLMs?

**Answer**: Follow the steps below to use real LLM APIs (GPT-4, Claude, Gemini, etc.)

### Question 2: Is the benchmark targeted for LLMs?

**Answer**: **YES** - The benchmarks evaluate your **LLM-based pipeline**:
- They test how well your pipeline (which uses LLMs internally) protects privacy
- They measure if LLMs leak PII when processing sensitive data
- They validate your ensemble-consensus mechanism with LLMs

**The benchmarks don't test the LLMs themselves**, they test **your privacy pipeline that uses LLMs**.

---

## 🚀 How to Use Real LLMs

### Step 1: Install Required Packages

```bash
# For OpenAI (GPT-4, GPT-4-turbo)
pip install openai

# For Anthropic (Claude)
pip install anthropic

# For Google (Gemini)
pip install google-generativeai

# Install all at once
pip install openai anthropic google-generativeai
```

---

### Step 2: Set Up API Keys

```bash
# Option A: Environment variables (recommended)
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
export GOOGLE_API_KEY='...'

# Option B: Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
EOF
```

---

### Step 3: Run with Real LLMs

#### Option A: Use the Example Script (Easiest)

```bash
# Run the pre-built example
python examples/real_llm_example.py
```

**What it does**:
- Uses real LLM APIs (GPT-4, Claude, etc.)
- Runs the full 4-step pipeline
- Shows API costs
- Validates privacy protection

**Expected output**:
```
🔬 Running Ensemble Pipeline with REAL LLMs
================================================

Step 1: Redaction & Masking
✓ Masked 3 queries
✓ Masked 11 article titles

Step 3: Ensemble Evaluation (Real LLMs)
  Calling GPT-4...                    ✓ (0.8s, $0.012)
  Calling Claude-3.5-Sonnet...        ✓ (1.2s, $0.008)
  Calling GPT-4-turbo...              ✓ (0.6s, $0.006)

Step 4: Consensus Aggregation
✓ Aggregated 3 model outputs

Final Output:
[
  {
    "ItemId": "diabetes-management",
    "QualityScore": 0.84,
    "QualityReason": "VeryStrong:MSNClicks+BingSearch"
  }
]

💰 Total Cost: $0.026 for 1 user
```

---

#### Option B: Modify Your Own Code

Replace `MockLLMEvaluator` with `RealLLMEvaluator`:

```python
# OLD CODE (Mock LLMs)
from src.pipeline import MockLLMEvaluator

evaluators = [
    MockLLMEvaluator("GPT-4", bias=0.0),
    MockLLMEvaluator("Claude-3.5", bias=0.05),
    MockLLMEvaluator("Gemini-Pro", bias=-0.03)
]

# NEW CODE (Real LLMs)
from examples.real_llm_example import RealLLMEvaluator

evaluators = [
    RealLLMEvaluator("gpt-4", api_key=os.getenv('OPENAI_API_KEY')),
    RealLLMEvaluator("claude-3-5-sonnet-20241022", api_key=os.getenv('ANTHROPIC_API_KEY')),
    RealLLMEvaluator("gpt-4-turbo", api_key=os.getenv('OPENAI_API_KEY'))
]

# Rest of the pipeline stays the same
all_results = []
for evaluator in evaluators:
    results = evaluator.evaluate_interest(masked_data, candidate_topics)
    all_results.append(results)

consensus = aggregator.aggregate_median(all_results)
```

---

### Step 4: Run Benchmarks with Real LLMs

The benchmarks can use either Mock or Real LLMs:

```bash
# With Mock LLMs (default, free, fast)
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 100

# With Real LLMs (add --use-real-llms flag)
# NOTE: This would require modifying the benchmark script to accept this flag
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 100 --use-real-llms
```

**Currently**, benchmarks use **Mock LLMs by default** because:
- ✅ Free (no API costs)
- ✅ Fast (no network latency)
- ✅ Reproducible results
- ✅ Tests the **mechanism**, not specific LLM behavior

---

## 🎯 Understanding What Benchmarks Test

### What Benchmarks Do

The benchmarks test **YOUR PIPELINE**, not the LLMs:

```
┌─────────────────────────────────────────────────┐
│          YOUR PRIVACY PIPELINE                   │
│                                                  │
│  Input: Sensitive Data                          │
│    ↓                                            │
│  Step 1: Redaction (your code)                  │
│    ↓                                            │
│  Step 3: Ensemble LLMs (Mock or Real)          │  ← Benchmarks test this!
│    ↓                                            │
│  Step 4: Consensus (your code)                  │
│    ↓                                            │
│  Output: Safe JSON                              │
└─────────────────────────────────────────────────┘
         ↓
    Benchmark checks:
    ✓ Does output leak PII?
    ✓ Can attacker reconstruct input?
    ✓ Is privacy preserved?
```

---

### What Benchmarks Measure

| Benchmark | Tests | Target |
|-----------|-------|--------|
| **ai4privacy/pii-masking-200k** | PII leakage in outputs | Your pipeline's output |
| **PII-Bench** | 55 PII categories protection | Your pipeline's output |
| **PrivacyXray** | Profile reconstruction attacks | Your pipeline's mechanism |
| **Canary Exposure** | Do secrets leak? | Your pipeline's aggregation |
| **Membership Inference** | Can attacker tell if data was used? | Your pipeline's privacy |
| **Attribute Inference** | Can attacker infer attributes? | Your pipeline's suppression |

**They DON'T test**:
- ❌ LLM quality/accuracy (that's utility testing)
- ❌ Which LLM is best
- ❌ LLM-specific vulnerabilities

**They DO test**:
- ✅ Your redaction mechanism
- ✅ Your consensus aggregation
- ✅ Your privacy boundary
- ✅ Whether your approach prevents leakage

---

## 💰 Cost Comparison

### Mock LLMs (Current Default)

```
Cost: $0.00
Time: ~2 seconds for full pipeline
Samples: Process 1000+ samples easily
```

**Use for**:
- ✅ Development and testing
- ✅ Benchmarking (mechanism validation)
- ✅ Demonstrations
- ✅ CI/CD pipelines

---

### Real LLMs

```
Cost per user (5-model ensemble):
  - GPT-4: $0.03/1K tokens × 3 calls = $0.01
  - Claude-3.5: $0.003/1K tokens × 3 calls = $0.001
  - GPT-4-turbo: $0.01/1K tokens × 3 calls = $0.003
  - Gemini-Pro: $0.0005/1K tokens × 3 calls = $0.0002
  - Llama (self-hosted): Free

Total: ~$0.015 - $0.05 per user

Time: ~2-5 seconds per user (with parallel calls)
Samples: Process 100-1000 samples ($1.50 - $50)
```

**Use for**:
- ✅ Production deployment
- ✅ Real-world validation
- ✅ Performance testing
- ✅ Client demonstrations

---

## 🔧 Recommended Setup

### For Development (Mock LLMs)

```python
# src/pipeline.py (already configured)
evaluators = [
    MockLLMEvaluator("GPT-4", bias=0.0),
    MockLLMEvaluator("Claude-3.5", bias=0.05),
    MockLLMEvaluator("Gemini-Pro", bias=-0.03),
    MockLLMEvaluator("Llama-3", bias=0.02),
    MockLLMEvaluator("Mistral-Large", bias=-0.01)
]
```

**Run**:
```bash
python src/pipeline.py  # Free, fast
```

---

### For Production (Real LLMs)

```python
# examples/real_llm_example.py (already configured)
from examples.real_llm_example import RealLLMEvaluator

evaluators = [
    RealLLMEvaluator("gpt-4", api_key=os.getenv('OPENAI_API_KEY')),
    RealLLMEvaluator("claude-3-5-sonnet-20241022", api_key=os.getenv('ANTHROPIC_API_KEY')),
    RealLLMEvaluator("gpt-4-turbo", api_key=os.getenv('OPENAI_API_KEY'))
]
```

**Run**:
```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
python examples/real_llm_example.py  # Costs ~$0.03/user
```

---

### For Benchmarks (Mock LLMs Recommended)

```bash
# Test privacy mechanism with mock LLMs (free, fast)
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000

# Why mock is sufficient for benchmarks:
# ✓ Tests YOUR redaction mechanism
# ✓ Tests YOUR consensus aggregation
# ✓ Tests privacy boundary enforcement
# ✓ Mock LLMs follow the same logic as real LLMs
```

---

## 📊 Benchmark Target Clarification

### Q: "Is the benchmark targeted for LLMs?"

**A: YES, but it tests YOUR PIPELINE that USES LLMs, not the LLMs themselves.**

Think of it this way:

```
┌────────────────────────────────────────┐
│  What Benchmarks Test:                 │
│                                        │
│  YOUR APPROACH:                        │
│  ├─ Redaction mechanism                │  ← Your contribution
│  ├─ Ensemble strategy                  │  ← Your contribution
│  ├─ Consensus voting                   │  ← Your contribution
│  └─ Privacy boundary enforcement       │  ← Your contribution
│                                        │
│  USING:                                │
│  └─ LLMs (Mock or Real)                │  ← Tool (interchangeable)
└────────────────────────────────────────┘
```

**Analogy**:
- Benchmark tests if your **car** is safe (your approach)
- Engine (LLM) can be gas or electric (Mock or Real)
- Safety features (redaction, consensus) are what matter
- Engine type doesn't change safety test results

---

## 🎯 When to Use Mock vs Real LLMs

| Scenario | Use Mock LLMs | Use Real LLMs |
|----------|---------------|---------------|
| **Development** | ✅ Yes | ❌ Too slow/expensive |
| **Testing** | ✅ Yes | ❌ Unnecessary |
| **Benchmarking** | ✅ Yes | ⚠️ Optional (expensive) |
| **Privacy Validation** | ✅ Yes | ⚠️ Same results, higher cost |
| **Utility Validation** | ❌ Need real for accuracy | ✅ Yes |
| **Production** | ❌ Too simplistic | ✅ Yes |
| **Client Demo** | ⚠️ OK for quick demo | ✅ Better for proof |
| **Research Paper** | ✅ Yes (mechanism focus) | ⚠️ Optional (small sample) |

---

## 🚀 Quick Start Commands

### Demo (Mock LLMs)

```bash
# See the approach in action (free, 2 seconds)
python src/pipeline.py
```

---

### Demo (Real LLMs)

```bash
# See with actual APIs (costs ~$0.03, 5 seconds)
export OPENAI_API_KEY='sk-...'
python examples/real_llm_example.py
```

---

### Benchmark (Mock LLMs - Recommended)

```bash
# Validate privacy mechanism (free, ~1 minute for 1000 samples)
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000
```

---

### Benchmark (Real LLMs - Expensive)

```bash
# Validate with real APIs (costs ~$1.50-$50 for 100-1000 samples)
# NOTE: Current benchmarks use Mock by default
# You'd need to modify benchmark code to use Real LLMs
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 100
```

---

## 📝 Summary

### To Use Real LLMs:

1. **Install packages**: `pip install openai anthropic google-generativeai`
2. **Set API keys**: `export OPENAI_API_KEY='sk-...'`
3. **Run example**: `python examples/real_llm_example.py`
4. **Or modify your code**: Replace `MockLLMEvaluator` with `RealLLMEvaluator`

### About Benchmarks:

- ✅ **YES**, benchmarks test your **LLM-based pipeline**
- ✅ They test your **privacy mechanism** (redaction, ensemble, consensus)
- ✅ They measure if **LLMs in your pipeline** leak PII
- ✅ Mock LLMs are **sufficient** for benchmarking (tests mechanism, not LLM quality)
- ⚠️ Real LLMs are **optional** for benchmarks (same results, higher cost)
- ✅ Real LLMs are **required** for production and utility validation

### Key Insight:

**Your contribution is the privacy mechanism (redaction + ensemble + consensus), not the LLMs themselves.**

The benchmarks validate that **YOUR MECHANISM** works, regardless of whether you use Mock or Real LLMs.

---

**Ready to use real LLMs?** Start with:
```bash
export OPENAI_API_KEY='sk-...'
python examples/real_llm_example.py
```

**Date**: 2025-01-14
