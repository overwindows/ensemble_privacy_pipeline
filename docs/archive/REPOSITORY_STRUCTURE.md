# Repository Structure

Complete file organization for the Ensemble-Redaction Privacy Pipeline repository.

---

## 📁 Directory Structure

```
ensemble-privacy-pipeline/
│
├── README.md                          # Main repository overview
├── QUICKSTART.md                      # 5-minute quick start guide
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore patterns
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
│
├── ensemble_privacy_pipeline.py      # ⭐ Main demo (mock LLMs)
├── ensemble_with_real_llms.py        # Production code (real APIs)
├── privacy_leakage_comparison.py     # ⭐ Privacy proof demo
│
├── docs/
│   ├── ENSEMBLE_PIPELINE_EXPLAINED.md # Technical deep dive
│   ├── README_ENSEMBLE_PIPELINE.md    # Comprehensive guide
│   └── PRIVACY_LEAKAGE_DEMO.md        # Leakage comparison analysis
│
├── examples/
│   ├── example_user_data.json         # Sample input data
│   └── example_output.json            # Sample output
│
└── tests/                              # Unit tests (to be added)
    ├── test_redactor.py
    ├── test_evaluator.py
    └── test_consensus.py
```

---

## 📄 File Descriptions

### Root Level

| File | Purpose | Key Features |
|------|---------|--------------|
| **README.md** | Main landing page | Overview, results, quick start, architecture |
| **QUICKSTART.md** | Get started in 5 min | Installation, first examples, basic usage |
| **CONTRIBUTING.md** | Contributor guide | Code standards, PR process, testing |
| **LICENSE** | MIT License | Open source terms |
| **.gitignore** | Git exclusions | Python, IDE, secrets, data files |
| **requirements.txt** | Dependencies | numpy, scikit-learn, openai, anthropic |
| **setup.py** | Package setup | Installation, entry points, metadata |

---

### Core Scripts

#### `ensemble_privacy_pipeline.py` ⭐ **START HERE**

**Purpose**: Complete working demo with mock LLMs

**Features**:
- Full 4-step pipeline implementation
- Privacy analysis included
- **No API keys needed**
- Runs in ~2 seconds

**Classes**:
- `PrivacyRedactor`: Masks sensitive data
- `MockLLMEvaluator`: Simulates LLM evaluation
- `ConsensusAggregator`: Aggregates results

**Run**:
```bash
python ensemble_privacy_pipeline.py
```

**Output**:
- Shows raw data → masked data → ensemble → consensus
- Privacy metrics (0% leakage)
- Utility metrics (100% preserved)

---

#### `ensemble_with_real_llms.py`

**Purpose**: Production code for real LLM APIs

**Features**:
- OpenAI (GPT-4, GPT-4-turbo)
- Anthropic (Claude-3.5-Sonnet)
- Google (Gemini) - partial
- Async parallel evaluation
- Cost estimation

**Classes**:
- `RealLLMEvaluator`: Calls real APIs
- `AsyncEnsembleEvaluator`: Parallel processing

**Run**:
```bash
export OPENAI_API_KEY='sk-...'
python ensemble_with_real_llms.py
```

**Cost**: ~$0.05/user for 5-model ensemble

---

#### `privacy_leakage_comparison.py` ⭐ **DEMO THIS**

**Purpose**: Prove the value of the pipeline

**Features**:
- Side-by-side comparison
- WITHOUT protection: 14 leaks
- WITH protection: 0 leaks
- Reconstruction attack simulation

**Classes**:
- `NoPrivacyEvaluator`: Unsafe baseline
- `analyze_privacy_leakage()`: Leak detection
- `reconstruction_attack()`: Adversarial testing

**Run**:
```bash
python privacy_leakage_comparison.py
```

**Output**:
```
╔═══════════════════════════════════════╦════════════════════╦════════════════════╗
║ Metric                                ║ Without Protection ║ With Protection    ║
╠═══════════════════════════════════════╬════════════════════╬════════════════════╣
║ Queries Leaked                        ║         3          ║         0          ║
║ Reconstruction Attack Success         ║        True        ║       False        ║
╚═══════════════════════════════════════╩════════════════════╩════════════════════╝
```

---

### Documentation (`docs/`)

#### `ENSEMBLE_PIPELINE_EXPLAINED.md`

**Purpose**: Technical deep dive

**Contents**:
- Step-by-step explanation
- Privacy analysis (how it prevents leakage)
- Comparison to formal DP
- When to use this approach
- Code structure walkthrough

**Read if**: You want to understand the "why" behind every decision

---

#### `README_ENSEMBLE_PIPELINE.md`

**Purpose**: Comprehensive reference guide

**Contents**:
- Quick start
- Performance metrics
- 4-step pipeline details
- Experiment protocols (I, II, III)
- Scaling considerations
- Security recommendations

**Read if**: You're deploying to production

---

#### `PRIVACY_LEAKAGE_DEMO.md`

**Purpose**: Detailed analysis of leakage comparison

**Contents**:
- Quantitative results
- Real-world impact
- Technical analysis
- ROI calculation
- Stakeholder presentation guide

**Read if**: You need to justify the approach to leadership

---

### Examples (`examples/`)

#### `example_user_data.json`

**Sample input**: User with diabetes, searches medical content

**Structure**:
```json
{
  "MSNClicks": [...],
  "BingSearch": [...],
  "demographics": {...},
  "candidate_topics": [...]
}
```

**Use**: Test your own modifications

---

#### `example_output.json`

**Sample output**: Safe JSON after pipeline

**Structure**:
```json
{
  "results": [
    {"ItemId": "A", "QualityScore": 0.85, "QualityReason": "MSNClicks+BingSearch"}
  ],
  "privacy_metrics": {...},
  "utility_metrics": {...}
}
```

**Use**: Validate your output format

---

### Tests (`tests/`)

**Status**: To be added (contributions welcome!)

**Planned**:
- `test_redactor.py`: Test masking and filtering
- `test_evaluator.py`: Test scoring logic
- `test_consensus.py`: Test aggregation methods
- `test_privacy.py`: Test leak detection
- `test_integration.py`: End-to-end tests

**Run**:
```bash
pytest tests/
```

---

## 🎯 Usage Paths

### For Quick Demo (5 minutes)

```
1. Clone repo
2. pip install -r requirements.txt
3. python ensemble_privacy_pipeline.py
```

**Result**: See full pipeline working with mock LLMs

---

### For Stakeholder Presentation (10 minutes)

```
1. Run: python privacy_leakage_comparison.py
2. Show output to stakeholders
3. Highlight: 14 leaks → 0 leaks
```

**Result**: Clear justification for adopting the approach

---

### For Production Deployment (1-2 days)

```
1. Read: docs/README_ENSEMBLE_PIPELINE.md
2. Set API keys for real LLMs
3. Modify: ensemble_with_real_llms.py for your data
4. Test on small dataset (10-100 users)
5. Measure privacy metrics
6. Scale up
```

**Result**: Production-ready deployment

---

### For Research/Experimentation (1 week)

```
1. Read: docs/ENSEMBLE_PIPELINE_EXPLAINED.md
2. Run all 3 experiments (masking, ensemble, consensus)
3. Modify scoring logic
4. Add new consensus methods
5. Publish results
```

**Result**: Novel contributions to the approach

---

## 🚀 Getting Started Checklist

- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Run basic demo (`python ensemble_privacy_pipeline.py`)
- [ ] Run leakage comparison (`python privacy_leakage_comparison.py`)
- [ ] Read quick start (`QUICKSTART.md`)
- [ ] Set up API keys (optional)
- [ ] Test with real LLMs (optional)
- [ ] Read full documentation (`docs/`)
- [ ] Customize for your use case
- [ ] Deploy!

---

## 📊 File Metrics

| Category | Count | Total Size |
|----------|-------|-----------|
| Python scripts | 3 | ~55 KB |
| Documentation | 4 | ~48 KB |
| Config files | 4 | ~2 KB |
| Examples | 2 | ~2 KB |
| **Total** | **13** | **~107 KB** |

---

## 🔄 Update History

### v1.0.0 (Initial Release)

**Core Files**:
- ✅ Full pipeline implementation
- ✅ Real LLM integration
- ✅ Privacy leakage comparison
- ✅ Comprehensive documentation

**Missing** (PRs welcome):
- ⏳ Unit tests
- ⏳ Deployment guides (AWS, Azure, GCP)
- ⏳ Performance benchmarks
- ⏳ Additional LLM providers (Gemini, Mistral)

---

## 📝 Maintenance

### Regular Updates

- **Dependencies**: Update quarterly
- **LLM APIs**: Update when providers change
- **Documentation**: Update with each feature
- **Tests**: Add with each PR

### Versioning

- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features
- **Patch** (1.0.0 → 1.0.1): Bug fixes

---

## 💬 Questions?

- See [QUICKSTART.md](QUICKSTART.md) for fast answers
- See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help
- Open GitHub Issue for bugs
- Open GitHub Discussion for questions

---

**Repository ready for GitHub!** 🚀
