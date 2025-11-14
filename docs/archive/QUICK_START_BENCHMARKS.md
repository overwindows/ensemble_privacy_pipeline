# 🚀 Quick Start: Public Benchmark Evaluation

**30-second setup** for evaluating the privacy pipeline on industry benchmarks.

---

## ⚡ Installation (2 minutes)

```bash
# Step 1: Core dependencies (required)
pip install numpy scikit-learn

# Step 2: Benchmark datasets (recommended)
pip install datasets huggingface_hub
```

---

## ✅ Quick Test (30 seconds)

```bash
# Verify everything works
python test_benchmarks.py
```

**Expected**: `✅ ALL TESTS PASSED!`

---

## 📊 Run Benchmarks (5 minutes)

### Option 1: Quick Evaluation (100 samples)
```bash
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 100
```

### Option 2: Standard Evaluation (1000 samples)
```bash
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 1000
```

### Option 3: Comprehensive (All Benchmarks)
```bash
python benchmark_public_datasets.py --benchmark all --num_samples 500
```

---

## 📈 View Results

```bash
# Results automatically saved to:
cat public_benchmark_results.json

# Or specify custom output:
python benchmark_public_datasets.py --benchmark ai4privacy --output my_results.json
```

---

## 🎯 What You'll See

```
========================================================================
  RESULTS: AI4PRIVACY
========================================================================

📊 Privacy Metrics:
  ├─ PII Leakage:
  │  ├─ Baseline (no privacy):     85.0%
  │  ├─ With Privacy Pipeline:      0.0%
  │  └─ Improvement:               85.0%
  │
  └─ Reconstruction Attack Success:
     ├─ Baseline:                  75.0%
     ├─ With Privacy:               0.0%
     └─ Improvement:               75.0%

🎯 Verdict: ✅ EXCELLENT - Pipeline provides >80% privacy improvement
```

---

## 📚 Benchmarks Available

| Benchmark | Command | Samples | Description |
|-----------|---------|---------|-------------|
| **ai4privacy** | `--benchmark ai4privacy` | 200K+ | Real-world PII dataset (54 types) |
| **PII-Bench** | `--benchmark pii-bench` | Custom | Query-aware privacy (55 types) |
| **PrivacyXray** | `--benchmark privacyxray` | Custom | Synthetic individuals (16 types) |
| **All** | `--benchmark all` | Mixed | Run all benchmarks |

---

## 🔧 Troubleshooting

### Issue: Import errors
```bash
pip install numpy scikit-learn datasets huggingface_hub
```

### Issue: Hugging Face authentication
```bash
huggingface-cli login
# Or skip and use synthetic fallback
```

### Issue: Out of memory
```bash
# Use fewer samples
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 100
```

---

## 📖 Full Documentation

- **Detailed Guide**: [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)
- **Summary**: [BENCHMARKS_SUMMARY.md](BENCHMARKS_SUMMARY.md)
- **Main README**: [README.md](README.md)

---

## 💡 Pro Tips

1. **Start small**: Use `--num_samples 100` for quick tests
2. **Use fallback**: If Hugging Face fails, synthetic data works fine
3. **Save results**: Always specify `--output` for reproducibility
4. **Compare**: Run multiple times and compare JSON results

---

**That's it! You're ready to evaluate privacy protection on industry benchmarks.** 🎉
