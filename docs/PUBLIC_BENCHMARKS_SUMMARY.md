# Public Benchmark Results

## Performance Summary

| Benchmark | Samples | Metric | Proposed Approach | Baseline | Difference |
|-----------|---------|--------|-------------------|----------|------------|
| **[ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)** | 1,000 | Full Protection | 28.8% (288/1000) | - | - |
| | | PII Types Tested | 54 types | - | - |
| **PUPA** (Li et al., NAACL 2025) | 901 | Response Success | 100.0% (901/901) | 85.5% | +14.5% |
| | | Privacy Leakage | 18.8% (902/4806) | 7.5% | +11.3% |
| **TAB** (Pilán et al., ACL 2022) | 1,268 | Direct ID Protection | 99.9% (1267/1268) | - | - |
| | | Quasi ID Protection | 99.9% (3801/3804) | - | - |
| | | Overall PII Masking | 83.7% (5308/6340) | - | - |

---

## References

1. **ai4privacy/pii-masking-200k**: https://huggingface.co/datasets/ai4privacy/pii-masking-200k
2. **PUPA Dataset**: Li et al., "PAPILLON: PrivAcy Preservation from Internet-based and Local Language Model Ensembles", NAACL 2025
3. **TAB Dataset**: Pilán et al., "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization", ACL 2022 Findings
4. **PAPILLON Baseline**: Li et al., NAACL 2025 (Quality=85.5%, Leakage=7.5%)
