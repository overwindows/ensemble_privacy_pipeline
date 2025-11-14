# Migration Guide

## Overview

The repository has been reorganized for better clarity and maintainability. This guide helps you understand the changes and migrate your code.

---

## 🔄 What Changed

### Old Structure → New Structure

| Old Location | New Location | Status |
|-------------|--------------|--------|
| `ensemble_privacy_pipeline.py` | `src/pipeline.py` | ✅ Moved, old file kept for compatibility |
| `evaluation_framework.py` | `src/evaluators.py` | ✅ Moved, old file kept |
| `benchmark_public_datasets.py` | `benchmarks/public_datasets.py` | ✅ Moved, old file kept |
| `benchmark_dp_specific.py` | `benchmarks/dp_specific.py` | ✅ Moved, old file kept |
| `test_benchmarks.py` | `tests/test_benchmarks.py` | ✅ Moved, old file kept |
| `ensemble_with_real_llms.py` | `examples/real_llm_example.py` | ✅ Moved, old file kept |
| `privacy_leakage_comparison.py` | `examples/privacy_comparison.py` | ✅ Moved, old file kept |
| `run_benchmark_comparison.py` | `benchmarks/comparison.py` | ✅ Moved, old file kept |
| Multiple `.md` files | `docs/archive/` | ✅ Archived, consolidated into README.md |

### Documentation Changes

| Old Files | New Files | Notes |
|-----------|-----------|-------|
| `README.md` (old) | `README.md` (new, comprehensive) | Merged content from 14+ docs |
| `DP.md` | `docs/PROTOCOL.md` | Protocol specification |
| `CONTRIBUTING.md` | `docs/CONTRIBUTING.md` | Contribution guidelines |
| All other `.md` files | `docs/archive/` | Archived for reference |

---

## 📂 New Folder Structure

```
ensemble-privacy-pipeline/
├── README.md                      # ⭐ NEW: Comprehensive unified guide
├── MIGRATION_GUIDE.md             # ⭐ NEW: This file
├── requirements.txt               # Updated with all dependencies
├── setup.py                       # Updated for new structure
│
├── src/                           # ⭐ NEW: Core pipeline code
│   ├── __init__.py
│   ├── pipeline.py                # Main pipeline (was ensemble_privacy_pipeline.py)
│   └── evaluators.py              # Evaluation framework (was evaluation_framework.py)
│
├── benchmarks/                    # ⭐ NEW: All benchmarks
│   ├── __init__.py
│   ├── public_datasets.py         # Public benchmark integration
│   ├── dp_specific.py             # DP-specific tests
│   └── comparison.py              # Utility comparison
│
├── examples/                      # ⭐ NEW: Example scripts
│   ├── real_llm_example.py        # Real LLM API usage
│   └── privacy_comparison.py      # Privacy proof demo
│
├── tests/                         # ⭐ NEW: Test suite
│   ├── __init__.py
│   └── test_benchmarks.py         # Benchmark tests
│
├── docs/                          # Documentation
│   ├── PROTOCOL.md                # Protocol specification (was DP.md)
│   ├── CONTRIBUTING.md            # Contribution guidelines
│   └── archive/                   # Archived documentation
│       ├── BENCHMARK_VALIDATION.md
│       ├── BENCHMARK_CONFIRMATION.md
│       ├── PIPELINE_FIX_SUMMARY.md
│       ├── BENCHMARK_GUIDE.md
│       ├── BENCHMARKS_SUMMARY.md
│       ├── QUICK_START_BENCHMARKS.md
│       ├── DP_BENCHMARK_GUIDE.md
│       ├── DP_LLM_Summary.md
│       ├── DP_LLM_Quick_Reference.md
│       ├── QUICKSTART.md
│       └── REPOSITORY_STRUCTURE.md
│
└── results/                       # ⭐ NEW: For benchmark outputs
    └── .gitkeep
```

---

## 🔧 How to Migrate Your Code

### Option 1: No Changes Needed (Backward Compatible)

The old files are **still in place** for backward compatibility. Your existing code will continue to work:

```python
# Old imports (still work)
from ensemble_privacy_pipeline import PrivacyRedactor, ConsensusAggregator
from evaluation_framework import PrivacyEvaluator

# Old scripts (still work)
python ensemble_privacy_pipeline.py
python benchmark_public_datasets.py
```

### Option 2: Migrate to New Structure (Recommended)

Update your imports to use the new structure:

```python
# New imports (recommended)
from src.pipeline import PrivacyRedactor, ConsensusAggregator, MockLLMEvaluator
from src.evaluators import PrivacyEvaluator, UtilityEvaluator

# New scripts
python src/pipeline.py
python benchmarks/public_datasets.py
python examples/privacy_comparison.py
```

---

## 📝 Import Migration Examples

### Example 1: Basic Pipeline Usage

**Old Code**:
```python
from ensemble_privacy_pipeline import PrivacyRedactor, MockLLMEvaluator, ConsensusAggregator

redactor = PrivacyRedactor()
evaluator = MockLLMEvaluator("GPT-4")
aggregator = ConsensusAggregator()
```

**New Code** (recommended):
```python
from src.pipeline import PrivacyRedactor, MockLLMEvaluator, ConsensusAggregator

redactor = PrivacyRedactor()
evaluator = MockLLMEvaluator("GPT-4")
aggregator = ConsensusAggregator()
```

**Or** (even better, using package import):
```python
from src import PrivacyRedactor, MockLLMEvaluator, ConsensusAggregator

redactor = PrivacyRedactor()
evaluator = MockLLMEvaluator("GPT-4")
aggregator = ConsensusAggregator()
```

---

### Example 2: Evaluation Framework

**Old Code**:
```python
from evaluation_framework import PrivacyEvaluator, UtilityEvaluator

privacy_eval = PrivacyEvaluator()
utility_eval = UtilityEvaluator()
```

**New Code**:
```python
from src.evaluators import PrivacyEvaluator, UtilityEvaluator

privacy_eval = PrivacyEvaluator()
utility_eval = UtilityEvaluator()
```

---

### Example 3: Benchmarks

**Old Code**:
```bash
python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 1000
python benchmark_dp_specific.py --test canary --num_samples 100
```

**New Code**:
```bash
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 1000
python benchmarks/dp_specific.py --test canary --num_samples 100
```

---

## 🆕 New Features in Reorganization

### 1. Package-Style Imports

You can now use cleaner imports:

```python
# Import entire package
import src

# Use classes
redactor = src.PrivacyRedactor()

# Or import from package __init__.py
from src import PrivacyRedactor, ConsensusAggregator
```

### 2. Centralized Documentation

All documentation is now consolidated:
- **Main guide**: `README.md` (comprehensive, single source of truth)
- **Protocol spec**: `docs/PROTOCOL.md`
- **Archived docs**: `docs/archive/` (for reference only)

### 3. Organized Code Structure

- **Core logic**: `src/` (reusable components)
- **Benchmarks**: `benchmarks/` (evaluation scripts)
- **Examples**: `examples/` (demo scripts)
- **Tests**: `tests/` (test suite)

---

## 🧪 Testing After Migration

### Run Quick Tests

```bash
# Test core pipeline (new location)
python src/pipeline.py

# Test privacy comparison (new location)
python examples/privacy_comparison.py

# Test benchmarks (new location)
python benchmarks/public_datasets.py --benchmark ai4privacy --num_samples 10
```

### Verify Imports

```python
# Test new imports
python -c "from src import PrivacyRedactor, ConsensusAggregator; print('✅ Imports work!')"

# Test old imports (backward compatibility)
python -c "from ensemble_privacy_pipeline import PrivacyRedactor; print('✅ Old imports still work!')"
```

---

## 📋 Checklist for Migration

- [ ] Read this migration guide
- [ ] Understand new folder structure
- [ ] Test old scripts (verify backward compatibility)
- [ ] Update imports in your code (recommended but optional)
- [ ] Test updated code
- [ ] Update any documentation or README references
- [ ] Commit changes

---

## 🐛 Troubleshooting

### Issue: ImportError after migration

**Error**:
```
ImportError: No module named 'src.pipeline'
```

**Solution**:
```bash
# Make sure you're in the repository root
cd /path/to/ensemble-privacy-pipeline

# Install in development mode
pip install -e .
```

---

### Issue: Old scripts not found

**Error**:
```
FileNotFoundError: ensemble_privacy_pipeline.py not found
```

**Solution**:
The old files are still there for backward compatibility. If they're missing:
```bash
# Copy from new structure back to root (for compatibility)
cp src/pipeline.py ensemble_privacy_pipeline.py
cp src/evaluators.py evaluation_framework.py
```

---

### Issue: Path not found in benchmarks

**Error**:
```
ModuleNotFoundError: No module named 'ensemble_privacy_pipeline'
```

**Solution** (in `benchmarks/public_datasets.py`):
```python
# Old import (may fail from new location)
from ensemble_privacy_pipeline import PrivacyRedactor

# New import (works from any location)
import sys
sys.path.append('..')
from src.pipeline import PrivacyRedactor
```

Or better, install the package:
```bash
pip install -e .
```

---

## 🔄 Rollback Instructions

If you need to rollback to the old structure:

```bash
# Old files are still present, no rollback needed!
# Just use the old file paths:
python ensemble_privacy_pipeline.py
python benchmark_public_datasets.py
```

---

## 📞 Need Help?

- **GitHub Issues**: For migration problems
- **GitHub Discussions**: For questions
- **Documentation**: See `README.md` for comprehensive guide

---

## 🎯 Summary

### Key Changes:
1. ✅ Code organized into `src/`, `benchmarks/`, `examples/`, `tests/`
2. ✅ Documentation consolidated into single `README.md`
3. ✅ Old files kept for backward compatibility
4. ✅ New package-style imports available

### Migration Impact:
- **Backward compatible**: Old code still works without changes
- **Recommended**: Update imports to new structure
- **No breaking changes**: All functionality preserved

### Timeline:
- **Now**: Use either old or new structure
- **Future (v2.0)**: Old root-level files may be deprecated (with warning period)

---

**You're ready to use the reorganized repository!** 🚀

For full documentation, see [README.md](README.md).
