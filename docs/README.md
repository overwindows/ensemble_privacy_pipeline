# Documentation

This directory contains detailed documentation for the Ensemble-Redaction Privacy Pipeline.

---

## Documentation Files

### 📖 [ENSEMBLE_PIPELINE_EXPLAINED.md](ENSEMBLE_PIPELINE_EXPLAINED.md)
Complete walkthrough of the ensemble-redaction pipeline with concrete examples.

**Contents:**
- Step-by-step explanation of the 4-phase pipeline
- Concrete example: diabetes interest evaluation
- Redaction, ensemble evaluation, consensus aggregation
- Privacy vs. utility tradeoffs

**Audience:** Developers wanting to understand how the core pipeline works

---

### 📊 [BENCHMARKS.md](BENCHMARKS.md)
Comprehensive benchmark evaluation guide and results.

**Contents:**
- ⚠️ **Critical bugfix notice** (2025-01-19) - Previous results invalid
- Public dataset benchmarks (ai4privacy, PUPA, TAB)
- Privacy attack benchmarks (DP comparison)
- Synthetic benchmarks (Vendor-neutral)
- Setup instructions and cost estimates
- **Bugfix details:** Ensemble consensus implementation fix

**Audience:** Researchers evaluating the approach, users running benchmarks

---

### 🔍 [PUPA_ENSEMBLE_WALKTHROUGH.md](PUPA_ENSEMBLE_WALKTHROUGH.md)
Detailed trace of ensemble consensus implementation (note: references old approach, needs update).

**Contents:**
- Step-by-step execution flow for one sample
- How 4 models produce different responses
- Consensus aggregation mechanics
- Additional bug discovery (redaction key mismatch)

**Audience:** Advanced users wanting to understand ensemble mechanics in detail

---

### 🚀 [RERUN_PLAN.md](RERUN_PLAN.md)
Step-by-step plan for running benchmarks with the corrected ensemble consensus code.

**Contents:**
- Priority order (quick test → medium → full)
- Expected consensus behavior
- Validation checklist
- Post-run action items

**Audience:** Users ready to run benchmarks with corrected code

---

## Quick Navigation

| I want to... | Read this |
|--------------|-----------|
| **Understand how the pipeline works** | [ENSEMBLE_PIPELINE_EXPLAINED.md](ENSEMBLE_PIPELINE_EXPLAINED.md) |
| **Run benchmarks** | [BENCHMARKS.md](BENCHMARKS.md) |
| **Understand the bugfix and run benchmarks** | [BENCHMARKS.md](BENCHMARKS.md#critical-bugfix-ensemble-consensus-2025-01-19) |
| **Get started quickly** | [../README.md](../README.md) |

---

## Recent Updates

### 2025-01-19: Critical Bugfix - Ensemble Consensus
- **Issues Fixed:**
  1. Benchmarks only using first model (not ensemble)
  2. Invalid use of ground truth for output selection
- **Proper Fix:** Majority voting consensus (no ground truth during aggregation)
- **Impact:** Previous results invalid, benchmarks must be re-run

See [BENCHMARKS.md - Critical Bugfix](BENCHMARKS.md#critical-bugfix-ensemble-consensus-2025-01-19) for complete details.

---

## Contributing to Documentation

When adding new documentation:
1. Place detailed guides in `docs/`
2. Keep `../README.md` concise (quick start only)
3. Update this README with links to new docs
4. Use clear section headers and examples
