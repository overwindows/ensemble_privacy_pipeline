# Benchmark Re-run Plan

## Why Re-run?

All public dataset benchmarks (Text Masking, PUPA, TAB) were only using **single-model performance** (gpt-oss-120b), not ensemble consensus.

The bugfix now properly uses all 4 models with **majority voting** consensus (no ground truth used during selection).

---

## Priority Order

Run benchmarks in this order (from cheapest to most expensive):

### 1. **Quick Validation Test** (Recommended First)
Verify the fixes work before spending money on full runs.

```bash
# Small test: 10 samples each (cost: ~$5)
export LLM_API_KEY='your-key-here'

python3 benchmarks/public_datasets_simple.py --num-samples 10
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 10
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 10
```

**What to check:**
- ✅ Console shows: `📊 Ensemble (4 models): X unique responses, consensus: majority`
- ✅ Consensus type shown (unanimous, majority, or shortest_fallback)
- ✅ No errors/crashes

**Expected output example:**
```
Sample 1/10: ai4privacy_0
  PII entities: 5 types
  📊 Ensemble (4 models): 2 unique outputs, consensus: majority
  ✅ ALL PII MASKED (5 entities protected)
```

---

### 2. **Medium-Scale Test** (If validation passes)
Moderate sample size for statistically meaningful results.

```bash
# Medium: 100 samples each (cost: ~$50)
python3 benchmarks/public_datasets_simple.py --num-samples 100
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 100
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 100
```

**Time estimate:** ~2-3 hours total
**Cost estimate:** ~$40-60

---

### 3. **Full-Scale Benchmarks** (For publication/final results)
Match the original claimed sample sizes.

```bash
# Full scale (cost: ~$150-200)
python3 benchmarks/public_datasets_simple.py --num-samples 1000
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 901
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 1268
```

**Time estimate:** ~8-12 hours total
**Cost estimate:** ~$150-200

---

## Cost Breakdown

| Benchmark | Samples | Models | Total API Calls | Est. Cost (@$0.01/call) |
|-----------|---------|--------|----------------|------------------------|
| **Text Masking** | 1,000 | 4 | 4,000 | $40 |
| **PUPA** | 901 | 4 | 3,604 | $36 |
| **TAB** | 1,268 | 4 | 5,072 | $51 |
| **Total** | 3,169 | 4 | **12,676** | **~$127** |

*Actual cost depends on your API pricing and rate limits*

---

## Expected Results

### Prediction: Ensemble Should Improve Privacy

**Old (Single Model - BROKEN):**
```
Text Masking:  28.8% protection
PUPA:          81.2% protected
TAB:           99.9% protected
```

**New (Ensemble - FIXED):**
```
Text Masking:  40-60% protection      (↑ 2× better)
PUPA:          85-92% protected       (↑ modest improvement)
TAB:           99.9% protected        (↔ already near-perfect)
```

**Why ensemble helps:**
- Uses **majority voting** to filter outlier responses
- Conservative fallback (shortest text) when no consensus
- Reduces worst-case privacy failures through consensus

---

## Comparison Plan

After re-running, create comparison table:

| Benchmark | Old (Single) | New (Ensemble) | Improvement | Valid? |
|-----------|--------------|----------------|-------------|---------|
| Text Masking | 28.8% | **??.?%** | **+??%** | ✅ After re-run |
| PUPA | 81.2% | **??.?%** | **+??%** | ✅ After re-run |
| TAB | 99.9% | **??.?%** | **±0%** | ✅ After re-run |

---

## Pre-Run Checklist

Before spending money on benchmarks:

- [ ] API key is set: `export LLM_API_KEY='...'`
- [ ] Test with 10 samples first (verify fixes work)
- [ ] Check available API credits/quota
- [ ] Verify console shows ensemble selection messages
- [ ] Ensure sufficient disk space for results (results/*.json)
- [ ] Consider running during off-peak hours (if rate limits exist)

---

## During Run

Monitor for:
- ✅ Console shows `📊 Ensemble (4 models): X unique responses, consensus: TYPE` for each sample
- ✅ Consensus types varying (unanimous, majority, shortest_fallback)
- ✅ Progress bar/counter working
- ⚠️ Any errors or model failures
- ⚠️ Rate limit warnings

---

## Post-Run Actions

1. **Verify results saved:**
   ```bash
   ls -lh results/*.json
   ```

2. **Compare old vs. new:**
   - Extract metrics from JSON files
   - Create comparison table
   - Calculate improvement percentages

3. **Update documentation:**
   - Update `README.md` with NEW benchmark numbers
   - Update `docs/BENCHMARKS.md` table
   - Remove "INVALID" warnings
   - Add "Re-run on 2025-XX-XX" note

4. **Commit changes:**
   ```bash
   git add results/*.json
   git add README.md docs/BENCHMARKS.md
   git commit -m "Re-run benchmarks with fixed ensemble consensus

   - Text Masking: XX.X% (was 28.8% with single model)
   - PUPA: XX.X% (was 81.2% with single model)
   - TAB: XX.X% (was 99.9% with single model)

   All benchmarks now properly use ensemble consensus."
   ```

---

## If Results Are Worse Than Expected

**Scenario:** Ensemble doesn't improve privacy much (or makes it worse)

**Possible reasons:**
1. **High model correlation** - All 4 models produce similar outputs (need more diverse models)
2. **Detection accuracy** - PII leakage checker has high false negative rate
3. **Task mismatch** - Free-form text tasks hard to aggregate via majority voting
4. **Systematic prompt issues** - All models fail similarly due to prompt design

**What to do:**
- Analyze per-sample results to understand failure modes
- Check consensus distribution (how often majority vs fallback?)
- Consider trying different model combinations
- Document findings honestly (negative results are still valuable!)

---

## Success Criteria

**Minimum acceptable outcome:**
- ✅ Benchmarks run without errors
- ✅ Console shows ensemble consensus working (majority/unanimous/fallback)
- ✅ Consensus types vary across samples (not all unanimous)
- ✅ Privacy leakage rate ≤ single-model baseline (not worse!)

**Ideal outcome:**
- ✅ Privacy leakage reduced by 30-50%
- ✅ Clear evidence majority voting provides benefit
- ✅ Results support ensemble-redaction approach claims

---

## Quick Start (Copy-Paste)

```bash
# 1. Set API key
export LLM_API_KEY='your-actual-api-key-here'

# 2. Quick validation (10 samples, ~$5)
python3 benchmarks/public_datasets_simple.py --num-samples 10
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 10
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 10

# 3. If validation passes, run medium scale (100 samples, ~$50)
python3 benchmarks/public_datasets_simple.py --num-samples 100
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 100
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 100

# 4. Check results
ls -lh results/*.json
```

---

## Questions to Answer After Re-run

1. **Does ensemble improve privacy?** By how much?
2. **How often does majority consensus occur?** vs unanimous vs fallback
3. **Are model outputs diverse enough?** (high unanimous rate = low diversity)
4. **Is the 4× cost justified?** (does 4 models give enough benefit?)
5. **Should you use different models?** (more diverse = better consensus?)

These answers will determine if the ensemble approach is actually valuable!
