# Quick Start Guide

Get up and running with the Ensemble-Redaction Pipeline in **5 minutes**.

---

## 🚀 Installation

### Option 1: From Source (Recommended for development)

```bash
git clone https://github.com/yourusername/ensemble-privacy-pipeline.git
cd ensemble-privacy-pipeline
pip install -r requirements.txt
```

### Option 2: As a Package (Coming soon)

```bash
pip install ensemble-privacy-pipeline
```

---

## 💻 Run Your First Example

### 1. Basic Demo (No API Keys Needed)

```bash
python ensemble_privacy_pipeline.py
```

**What it does:**
- Loads example user data (medical queries)
- Masks sensitive information
- Evaluates with 5 mock LLM models
- Aggregates with consensus
- Shows privacy analysis

**Expected output:**
```
================================================================================
FINAL OUTPUT (EXITS PRIVACY BOUNDARY)
================================================================================

✓ Safe to release - contains ONLY:
  - ItemId (no user data)
  - QualityScore (aggregated, smoothed by ensemble)
  - QualityReason (generic source types only: MSNClicks+BingSearch)

JSON Output:
[
  {
    "ItemId": "A",
    "QualityScore": 0.85,
    "QualityReason": "VeryStrong:MSNUpvotes+MSNClicks+BingSearch"
  }
]
```

**Time:** ~2 seconds

---

### 2. Privacy Leakage Comparison

```bash
python privacy_leakage_comparison.py
```

**What it does:**
- Shows side-by-side comparison
- WITHOUT protection: 14 private data leaks
- WITH protection: 0 leaks
- Demonstrates reconstruction attack

**Expected output:**
```
╔═══════════════════════════════════════╦════════════════════╦════════════════════╗
║ Metric                                ║ Without Protection ║ With Protection    ║
╠═══════════════════════════════════════╬════════════════════╬════════════════════╣
║ Queries Leaked                        ║         3          ║         0          ║
║ Titles Leaked                         ║         11         ║         0          ║
║ Reconstruction Attack Success         ║        True        ║       False        ║
╚═══════════════════════════════════════╩════════════════════╩════════════════════╝
```

**Time:** ~3 seconds

---

## 🔑 Use Real LLM APIs

### 1. Set Up API Keys

```bash
# OpenAI (GPT-4)
export OPENAI_API_KEY='sk-...'

# Anthropic (Claude)
export ANTHROPIC_API_KEY='sk-ant-...'

# Google (Gemini)
export GOOGLE_API_KEY='...'
```

### 2. Run with Real Models

```bash
python ensemble_with_real_llms.py
```

**What it does:**
- Uses actual LLM APIs (GPT-4, Claude, etc.)
- Runs ensemble in parallel for speed
- Estimates costs

**Cost:** ~$0.05 per user (5-model ensemble)

---

## 🧑‍💻 Use in Your Code

### Basic Usage

```python
from ensemble_privacy_pipeline import PrivacyRedactor, ConsensusAggregator
from ensemble_with_real_llms import RealLLMEvaluator

# 1. Load your data
raw_user_data = {
    "MSNClicks": [...],
    "BingSearch": [...],
    "demographics": {...}
}

candidate_topics = [
    {"ItemId": "A", "Topic": "Managing diabetes"},
    {"ItemId": "B", "Topic": "AI news"}
]

# 2. Redact sensitive data
redactor = PrivacyRedactor()
masked_data = redactor.redact_user_data(raw_user_data)

# 3. Evaluate with ensemble
models = [
    RealLLMEvaluator("gpt-4", api_key="..."),
    RealLLMEvaluator("claude-3-5-sonnet-20241022", api_key="..."),
    RealLLMEvaluator("gpt-4-turbo", api_key="..."),
]

all_results = []
for model in models:
    results = model.evaluate_interest(masked_data, candidate_topics)
    all_results.append(results)

# 4. Aggregate with consensus
aggregator = ConsensusAggregator()
final_output = aggregator.aggregate_median(all_results)

# 5. Use safe output
print(final_output)
# [{"ItemId": "A", "QualityScore": 0.85, "QualityReason": "MSNClicks+BingSearch"}]
```

---

## 📊 Understand the Output

### Output Format

```json
{
  "ItemId": "A",
  "QualityScore": 0.85,
  "QualityReason": "VeryStrong:MSNClicks+BingSearch"
}
```

**Fields:**
- `ItemId`: Candidate topic identifier
- `QualityScore`: 0-1 float (higher = stronger interest)
- `QualityReason`: Evidence sources (GENERIC, not specific queries)

### Score Interpretation

| Score | Interpretation | Example Reason |
|-------|---------------|----------------|
| 0.85+ | Very strong evidence (3+ sources) | `VeryStrong:MSNClicks+BingSearch+Upvotes` |
| 0.70-0.82 | Strong evidence (2 sources) | `Strong:MSNClicks+BingSearch` |
| 0.55-0.70 | Moderate evidence (1 strong source) | `Moderate:MSNClicks` |
| 0.35-0.55 | Weak evidence (1 weak source) | `Weak:MAI` |
| <0.35 | No evidence | `no supporting evidence` |
| 0.20 | Demographic mismatch | `demographic mismatch` |

### What's Safe

✅ **Safe to release:**
- ItemId (just labels, no user data)
- Score (aggregated across models)
- Generic source types (`MSNClicks`, `BingSearch`)

❌ **Never in output:**
- Specific queries ("diabetes diet plan")
- Article titles ("New diabetes treatment...")
- URLs, timestamps, exact demographics
- Individual model predictions

---

## 🎯 Next Steps

### 1. Read Documentation

- **[Full Guide](docs/README_ENSEMBLE_PIPELINE.md)**: Comprehensive documentation
- **[Technical Deep Dive](docs/ENSEMBLE_PIPELINE_EXPLAINED.md)**: How it works
- **[Privacy Analysis](docs/PRIVACY_LEAKAGE_DEMO.md)**: Proof of effectiveness

### 2. Customize for Your Use Case

```python
# Customize masking
redactor = PrivacyRedactor()
# Add your own masking rules
redactor.custom_patterns = [...]

# Customize scoring tiers
evaluator = RealLLMEvaluator("gpt-4")
# Adjust thresholds
evaluator.score_tiers = {...}

# Customize consensus method
aggregator = ConsensusAggregator()
# Try different methods
consensus = aggregator.aggregate_intersection(all_results)  # Most conservative
consensus = aggregator.aggregate_trimmed_mean(all_results)  # Outlier-robust
```

### 3. Run Experiments

Follow the experimental protocol:

```bash
# Experiment I: Test masking strategies
python experiments/test_masking.py

# Experiment II: Test ensemble sizes
python experiments/test_ensemble.py

# Experiment III: Test consensus methods
python experiments/test_consensus.py
```

### 4. Deploy to Production

See deployment guides:
- **AWS**: `docs/deployment/aws.md`
- **Azure**: `docs/deployment/azure.md`
- **GCP**: `docs/deployment/gcp.md`

---

## 🐛 Troubleshooting

### Issue: API Key Not Found

```
Error: No valid API keys found
```

**Solution:**
```bash
export OPENAI_API_KEY='your-key-here'
# Or add to .env file
```

### Issue: Import Error

```
ImportError: No module named 'ensemble_privacy_pipeline'
```

**Solution:**
```bash
# Make sure you're in the repo directory
cd ensemble-privacy-pipeline

# Install in development mode
pip install -e .
```

### Issue: Rate Limiting

```
Error: Rate limit exceeded
```

**Solution:**
```python
# Add retry logic
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(min=1, max=60))
def call_llm(...):
    # Your API call
```

---

## 💡 Tips

1. **Start with mock LLMs** (free, fast)
2. **Test on small dataset** (1-10 users)
3. **Measure privacy** (run leakage comparison)
4. **Optimize costs** (use cheaper models in ensemble)
5. **Run in parallel** (for production speed)

---

## 📞 Get Help

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions
- **Email**: help@example.com

---

**You're ready to go!** 🚀

Run your first example:
```bash
python ensemble_privacy_pipeline.py
```
