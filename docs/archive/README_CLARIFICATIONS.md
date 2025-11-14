# README Clarifications

**Important Notes About the Numbers in README.md**

---

## ❓ Question 1: Which LLMs are used in evaluation?

### Answer: Mock LLMs (Simulated)

**Current Implementation** uses **MockLLMEvaluator** - simulated LLM behavior, NOT real API calls.

**Location**: [src/pipeline.py:191-367](src/pipeline.py#L191-L367)

```python
class MockLLMEvaluator:
    """
    Simulates an LLM interest evaluator.

    In production, this would call actual LLMs (GPT-4, Claude, Gemini, etc.)
    inside the privacy boundary.

    For this demo, we simulate different model behaviors.
    """
```

**5 Mock Models Used** (lines 591-595):
```python
MockLLMEvaluator("GPT-4", bias=0.0),
MockLLMEvaluator("Claude-3.5", bias=0.05),
MockLLMEvaluator("Gemini-Pro", bias=-0.03),
MockLLMEvaluator("Llama-3", bias=0.02),
MockLLMEvaluator("Mistral-Large", bias=-0.01)
```

**Why Mock?**
- ✅ No API keys needed
- ✅ No API costs
- ✅ Fast execution (~2 seconds)
- ✅ Demonstrates the pipeline logic
- ✅ Reproducible results

**For Real LLMs**: Use [examples/real_llm_example.py](examples/real_llm_example.py)

---

## ❓ Question 2: Where do the numbers in README.md come from?

### Answer: The numbers come from the demo script, NOT from running your pipeline

**The numbers in README.md are ILLUSTRATIVE**, taken from the demo script to show what the approach WOULD achieve.

### Specific Numbers & Their Sources

#### 1. Privacy Metrics Table (Lines 59-65)

```
| Metric | Without Protection | With Pipeline | Improvement |
|--------|-------------------|---------------|-------------|
| Queries Leaked | 3 (75%) | 0 (0%) | ✅ 100% |
| Titles Leaked | 11 (100%) | 0 (0%) | ✅ 100% |
| Medical Info Inferred | 6 conditions | 0 conditions | ✅ 100% |
| Reconstruction Attack | ✅ Success | ❌ Failed | ✅ 100% |
```

**Source**: [examples/privacy_comparison.py](examples/privacy_comparison.py)
- This is a **demo script** that runs the comparison
- Uses **example data** (4 queries, 11 article titles)
- Counts leaks in "without protection" vs "with protection"
- These are **demonstrative results**, not from running on your actual dataset

**To see these numbers yourself**:
```bash
python examples/privacy_comparison.py
```

---

#### 2. Benchmark Validation Table (Lines 67-74)

```
| Benchmark | PII Leakage | Reconstruction Rate | Status |
|-----------|-------------|---------------------|--------|
| ai4privacy/pii-masking-200k | 0.0% | 0.0% | ✅ Passed |
| PII-Bench (ACL 2024) | 0.0% | 0.0% | ✅ Passed |
| PrivacyXray (2025) | 0.0% | 0.0% | ✅ Passed |
| Canary Exposure (PrivLM-Bench) | 2.0% | N/A | ✅ DP-like |
| Membership Inference Attack | AUC 0.58 | N/A | ✅ DP-like |
```

**Source**: **EXPECTED RESULTS** based on benchmark design
- These benchmarks have been **integrated** in [benchmarks/](benchmarks/)
- The 0.0% numbers are **expected** because the pipeline outputs only generic metadata
- The 2.0% and AUC 0.58 are **typical values** from DP research literature

**Status**: ⚠️ **NOT YET RUN on your actual data**

**To run these benchmarks yourself**:
```bash
# Public benchmarks
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000

# DP-specific benchmarks
python benchmarks/dp_specific.py --test canary --num_samples 100
python benchmarks/dp_specific.py --test mia --num_samples 200
```

---

#### 3. Utility Metrics (Lines 76-81)

```
| Metric | Target | Achieved |
|--------|--------|----------|
| Score Accuracy | High | ✅ 0.85 (same as baseline) |
| Score Drift | ≤5% | ✅ 0% |
| Format Stability | 100% | ✅ 100% |
```

**Source**: **Example values** from demo script
- The 0.85 score is from the **mock evaluator logic**
- Score drift 0% is because mock evaluators are deterministic
- These are **illustrative** to show the approach maintains utility

---

## 🔍 Summary

| Numbers | Status | Source | How to Verify |
|---------|--------|--------|---------------|
| **Privacy Metrics** (3 queries, 11 titles) | ✅ Accurate for demo | `privacy_comparison.py` demo | Run `python examples/privacy_comparison.py` |
| **Benchmark Results** (0.0% leakage) | ⚠️ Expected, not measured | Benchmark design + literature | Run `python benchmarks/public_datasets.py` |
| **Utility Metrics** (0.85 score) | ✅ From mock LLMs | Mock evaluator logic | Run `python src/pipeline.py` |
| **DP Comparison** (2.0%, AUC 0.58) | ⚠️ Literature values | DP research papers | Run `python benchmarks/dp_specific.py` |

---

## 🎯 What You Should Do

### 1. Run the Demo to See Actual Results

```bash
# Quick demo (3 seconds)
python examples/privacy_comparison.py
```

**This will show you**:
- Exact leak counts for the example data
- Before/after comparison
- Reconstruction attack results

---

### 2. Run Benchmarks to Get Real Numbers

```bash
# Test on real-world datasets
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000

# Expected output:
# {
#   "privacy_metrics": {
#     "pii_leakage": {
#       "baseline": 0.85,      # 85% of baseline outputs leak PII
#       "with_privacy": 0.00,  # 0% with your approach
#       "improvement_pct": 85.0
#     }
#   }
# }
```

---

### 3. Update README with YOUR Actual Results

After running benchmarks, update the tables in README.md with:
- ✅ Your actual leak counts
- ✅ Your actual benchmark results
- ✅ Your actual DP comparison numbers

---

## 🔧 To Get Real Numbers

### Step 1: Run Privacy Comparison

```bash
python examples/privacy_comparison.py
```

**Output location**: Terminal output
**What you get**: Leak counts, reconstruction attack results

---

### Step 2: Run Public Benchmarks

```bash
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000
```

**Output location**: `results/benchmark_public_datasets_TIMESTAMP.json`
**What you get**: PII leakage rates, reconstruction rates

---

### Step 3: Run DP Benchmarks

```bash
python benchmarks/dp_specific.py --run_full_evaluation --num_samples 100
```

**Output location**: `results/benchmark_dp_specific_TIMESTAMP.json`
**What you get**: Canary exposure, MIA AUC, attribute inference rates

---

### Step 4: Update README.md

Replace the example numbers with your actual measured results.

---

## ⚠️ Important Clarifications

### Mock LLMs vs Real LLMs

**Current code uses MOCK LLMs**:
- ✅ Fast (no API calls)
- ✅ Free (no API costs)
- ✅ Reproducible
- ❌ Not using real GPT-4/Claude/Gemini

**To use REAL LLMs**:
```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
python examples/real_llm_example.py
```

---

### Numbers in README are EXPECTED, not MEASURED

The README was written to show:
1. ✅ What the approach is designed to achieve
2. ✅ Expected results based on the mechanism
3. ⚠️ NOT actual measured results on your specific data

**To get MEASURED results**: Run the benchmarks as described above.

---

## 📝 Recommended Updates to README

Consider adding this section to README.md:

```markdown
## ⚠️ Note About Results

The results shown in this README are:
- **Privacy Metrics**: Based on demo script with example data
- **Benchmark Results**: Expected results based on mechanism design
- **DP Comparison**: Typical values from research literature

**To reproduce these results**:
\`\`\`bash
# Run demo
python examples/privacy_comparison.py

# Run benchmarks
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000
python benchmarks/dp_specific.py --run_full_evaluation --num_samples 100
\`\`\`
```

---

## 🎯 Action Items

1. **Run `privacy_comparison.py`** to see the demo leak counts
2. **Run benchmarks** to get actual measurements on real datasets
3. **Update README.md** with measured results (not expected results)
4. **Clarify in README** whether numbers are measured or expected
5. **Consider adding a "Reproducibility" section** with exact commands to generate each number

---

**Key Takeaway**: The README numbers are **illustrative/expected**, not **measured**. Run the benchmarks to get actual numbers for your implementation.

**Date**: 2025-01-14
**Status**: Clarified
