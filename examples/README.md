# Examples & Demos

This folder contains working examples, demonstrations, and Jupyter notebooks for the Ensemble-Redaction Consensus Pipeline.

---

## 📂 Contents

### Python Scripts

| File | Description | Run Time | API Keys Needed |
|------|-------------|----------|-----------------|
| [`privacy_comparison.py`](privacy_comparison.py) | Privacy leakage demo (14 leaks → 0) | ~3 sec | ❌ No |
| [`real_llm_example.py`](real_llm_example.py) | Production code with real LLM APIs | ~10 sec | ✅ Yes |

### Jupyter Notebooks

| Notebook | Approach | Description | Key Topics |
|----------|----------|-------------|------------|
| [`Non_DP_Ensemble_Consensus_Pipeline.ipynb`](Non_DP_Ensemble_Consensus_Pipeline.ipynb) | **Non-DP (Our Approach)** | Working implementation of ensemble-consensus pipeline | Redaction, masking, ensemble, consensus |
| [`DP_Inference_Exploration_Challenges.ipynb`](DP_Inference_Exploration_Challenges.ipynb) | **DP Exploration** | Explores DP at inference time and its challenges | OpenDP, logit aggregation, DP limitations |

### Data Files

| File | Description |
|------|-------------|
| [`example_user_data.json`](example_user_data.json) | Sample input: User with medical queries |
| [`example_output.json`](example_output.json) | Sample output: Safe JSON after pipeline |

---

## 🎯 Quick Start

### Option 1: Privacy Comparison Demo (Recommended First)

**Shows dramatic before/after comparison**:

```bash
python privacy_comparison.py
```

**Output**:
```
╔═══════════════════════════════════════╦════════════════════╦════════════════════╗
║ Metric                                ║ Without Protection ║ With Protection    ║
╠═══════════════════════════════════════╬════════════════════╬════════════════════╣
║ Queries Leaked                        ║         3          ║         0          ║
║ Titles Leaked                         ║         11         ║         0          ║
║ Medical Info Inferred                 ║         6          ║         0          ║
║ Reconstruction Attack Success         ║        True        ║       False        ║
╚═══════════════════════════════════════╩════════════════════╩════════════════════╝
```

**Time**: ~3 seconds | **API Keys**: Not needed ✅

---

### Option 2: Real LLM APIs (Production)

**Uses actual LLM APIs (GPT-4, Claude, etc.)**:

```bash
# Set API keys
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'

# Run with real models
python real_llm_example.py
```

**Time**: ~10 seconds | **Cost**: ~$0.05/user

---

## 📓 Jupyter Notebooks

### 1. Non-DP Ensemble-Consensus Pipeline (Our Approach)

**File**: [`Non_DP_Ensemble_Consensus_Pipeline.ipynb`](Non_DP_Ensemble_Consensus_Pipeline.ipynb)

**What it demonstrates**:
- ✅ **Step 1: Redaction & Masking** - Replace queries with tokens
- ✅ **Step 3: Ensemble Evaluation** - Multiple LLM evaluators
- ✅ **Step 4: Consensus Aggregation** - Median + majority voting
- ✅ **Privacy Boundary** - Eyes-off data processing

**Approach**: **Non-DP (Training-Free)**

**Key Features**:
- Working implementation of the 4-step pipeline
- Mock LLM evaluators (no API keys needed)
- Shows redaction, ensemble, and consensus in action
- Demonstrates privacy boundary concept

**When to use**:
- Understand the ensemble-consensus approach
- Quick prototype without API costs
- Educational demonstrations
- Validate pipeline logic

**Run**:
```bash
jupyter notebook examples/Non_DP_Ensemble_Consensus_Pipeline.ipynb
```

---

### 2. DP Inference Exploration & Challenges

**File**: [`DP_Inference_Exploration_Challenges.ipynb`](DP_Inference_Exploration_Challenges.ipynb)

**What it demonstrates**:
- 🔬 **Differential Privacy (DP)** at inference time
- 🔬 **Logit Aggregation** with Laplace noise
- ⚠️ **Challenges of DP inference** for text generation
- 🔬 **OpenDP library** usage
- 📊 **Privacy-utility tradeoff** exploration

**Approach**: **Differential Privacy (DP)**

**Key Features**:
- Uses OpenDP library for formal DP guarantees
- Explores DP aggregation for text generation
- Discusses fundamental challenges:
  - High dimensionality (50k-100k vocabulary)
  - Query clustering requirements
  - Utility degradation with small groups
  - Privacy-utility tradeoff
- Explains why DP-SGD (training-time DP) is preferred in practice

**When to use**:
- Understand formal Differential Privacy
- Explore DP at inference time
- Learn DP limitations for text generation
- Compare DP vs. Non-DP approaches

**Run**:
```bash
# Install dependencies
pip install opendp torch transformers

jupyter notebook examples/DP_Inference_Exploration_Challenges.ipynb
```

---

## 🆚 Comparison: DP vs Non-DP Notebooks

| Aspect | Non-DP Ensemble-Consensus | DP Inference Exploration |
|--------|---------------------------|--------------------------|
| **Approach** | Ensemble + Consensus (Training-Free) | Differential Privacy (Formal) |
| **Privacy Method** | Input masking + voting | Noise addition (Laplace) |
| **Privacy Guarantee** | Empirical (benchmark-validated) | Formal (mathematical proof) |
| **Utility** | ✅ High (0% degradation) | ⚠️ Lower (noise impact) |
| **Training Needed** | ❌ No | ⚠️ Prefers DP-SGD (training) |
| **API Cost** | ~$0.05/user (5 models) | Variable |
| **Complexity** | Simple (redact → ensemble → vote) | Complex (logit aggregation) |
| **Production Ready** | ✅ Yes (implemented in src/) | ⚠️ Exploration only |
| **Use Case** | API-only scenarios, fast deployment | Research, formal guarantees |
| **Notebook Purpose** | Working implementation | Exploration & education |

---

## 📊 Example Data

### `example_user_data.json`

Sample input representing a user with diabetes-related interests:

```json
{
  "MSNClicks": [
    {"title": "Understanding Type 2 Diabetes: Symptoms and Prevention", "timestamp": "2024-01-15T10:00:00"},
    {"title": "Diabetes Diet Plan: Foods to Eat and Avoid", "timestamp": "2024-01-15T11:00:00"}
  ],
  "BingSearch": [
    {"query": "diabetes symptoms", "timestamp": "2024-01-15T09:00:00"},
    {"query": "diabetes diet plan", "timestamp": "2024-01-15T09:30:00"}
  ],
  "demographics": {
    "age": 45,
    "gender": "F",
    "location": "Seattle"
  }
}
```

---

### `example_output.json`

Safe output after privacy pipeline (no PII leaked):

```json
{
  "results": [
    {
      "ItemId": "diabetes-management",
      "QualityScore": 0.85,
      "QualityReason": "VeryStrong:MSNClicks+BingSearch"
    }
  ],
  "privacy_metrics": {
    "pii_leakage": "0%",
    "reconstruction_attack": "Failed"
  }
}
```

**Notice**: Only generic source types (`MSNClicks`, `BingSearch`), no specific queries!

---

## 🚀 Usage Patterns

### Pattern 1: Quick Demo for Stakeholders

```bash
# Show dramatic privacy improvement
python privacy_comparison.py
```

**Time**: 3 seconds
**Impact**: Show 14 leaks → 0 leaks

---

### Pattern 2: Understand the Approach

```bash
# Open Non-DP notebook
jupyter notebook examples/Non_DP_Ensemble_Consensus_Pipeline.ipynb
```

**Goal**: Understand redaction, ensemble, consensus
**Time**: 15 minutes

---

### Pattern 3: Explore DP Alternative

```bash
# Open DP notebook
jupyter notebook examples/DP_Inference_Exploration_Challenges.ipynb
```

**Goal**: Understand DP approach and its challenges
**Time**: 30 minutes

---

### Pattern 4: Production Deployment

```bash
# Test with real APIs
export OPENAI_API_KEY='sk-...'
python real_llm_example.py
```

**Goal**: Validate approach with real LLMs
**Cost**: ~$0.05 per user

---

## 📚 Additional Resources

### Related Documentation

- **[Main README](../README.md)** - Comprehensive guide
- **[Protocol Spec](../docs/PROTOCOL.md)** - 4-step pipeline specification
- **[Pipeline Code](../src/pipeline.py)** - Production implementation

### Related Scripts

- **[Benchmarks](../benchmarks/)** - Validation on 200K+ samples
- **[Tests](../tests/)** - Test suite

---

## 🎓 Learning Path

**If you're new**, follow this order:

1. **Start**: `python privacy_comparison.py` (3 sec - see the problem)
2. **Understand**: Open `Non_DP_Ensemble_Consensus_Pipeline.ipynb` (15 min - see the solution)
3. **Compare**: Open `DP_Inference_Exploration_Challenges.ipynb` (30 min - see alternatives)
4. **Deploy**: `python real_llm_example.py` (10 sec - test with real APIs)

**Total time**: ~1 hour to full understanding

---

## 🐛 Troubleshooting

### Issue: Jupyter Not Installed

```bash
pip install jupyter notebook
```

### Issue: OpenDP Import Error (DP notebook)

```bash
pip install opendp
```

### Issue: API Keys Not Found (real_llm_example.py)

```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
```

### Issue: Module Not Found

```bash
# Make sure you're in repo root
cd /path/to/ensemble-privacy-pipeline

# Install in development mode
pip install -e .
```

---

## 📞 Questions?

- **Main README**: [../README.md](../README.md)
- **GitHub Issues**: For bugs and questions
- **Protocol Spec**: [../docs/PROTOCOL.md](../docs/PROTOCOL.md)

---

## 🎯 Summary

| File | Type | Approach | Purpose | Run Time |
|------|------|----------|---------|----------|
| `privacy_comparison.py` | Script | Non-DP | Privacy demo (14→0 leaks) | 3 sec |
| `real_llm_example.py` | Script | Non-DP | Production with real APIs | 10 sec |
| `Non_DP_Ensemble_Consensus_Pipeline.ipynb` | Notebook | **Non-DP** | Working pipeline implementation | Interactive |
| `DP_Inference_Exploration_Challenges.ipynb` | Notebook | **DP** | DP exploration & challenges | Interactive |
| `example_user_data.json` | Data | - | Sample input | - |
| `example_output.json` | Data | - | Sample output | - |

**All examples are ready to run!** 🚀
