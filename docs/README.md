# Documentation Index

Welcome to the Ensemble-Redaction Privacy Pipeline documentation!

## 📚 Quick Start

- **[Main README](../README.md)** - Start here! Overview, installation, and quick start
- **[Pipeline Explained](ENSEMBLE_PIPELINE_EXPLAINED.md)** - How the ensemble-redaction approach works
- **[Privacy Leakage Demo](PRIVACY_LEAKAGE_DEMO.md)** - Visual demonstration of privacy protection

## 🔧 Implementation Guides

- **[Scripts Summary](SCRIPTS_SUMMARY.md)** - Complete guide to all scripts and benchmarks
- **[Guides](GUIDES.md)** - Step-by-step setup and usage guides
- **[Protocol](PROTOCOL.md)** - Technical protocol specification

## 🛠️ Technical Details

### Critical Issues & Fixes

- **[Alignment Analysis](ALIGNMENT_ANALYSIS.md)** ⚠️ **IMPORTANT**
  - Documents the critical misalignment issue (now fixed)
  - Shows why PrivacyRedactor was failing with public datasets
  - Verification steps and fix details

- **[Field Names Comparison](FIELD_NAMES_COMPARISON.md)**
  - Microsoft-specific vs vendor-neutral field names
  - Why vendor-neutral names matter for research
  - Migration guide for existing code

### Development

- **[Changelog](CHANGELOG.md)** - Version history and updates
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project

## 📊 Benchmark Documentation

See [benchmarks/README.md](../benchmarks/README.md) for:
- Available benchmark scripts
- Public dataset integrations
- Usage instructions and cost estimates

## 🗂️ Archive

Older documentation is preserved in [archive/](archive/) for reference.

---

## Documentation Organization

```
docs/
├── README.md                           # This file - documentation index
├── ENSEMBLE_PIPELINE_EXPLAINED.md      # Core concepts and architecture
├── PRIVACY_LEAKAGE_DEMO.md             # Visual privacy demo
├── SCRIPTS_SUMMARY.md                  # All scripts reference
├── GUIDES.md                           # Setup and usage guides
├── PROTOCOL.md                         # Technical specification
├── ALIGNMENT_ANALYSIS.md               # Critical fix documentation ⚠️
├── FIELD_NAMES_COMPARISON.md           # Field naming guide
├── CHANGELOG.md                        # Version history
├── CONTRIBUTING.md                     # Contribution guidelines
└── archive/                            # Older documentation
```

## 🎯 Recommended Reading Order

### For New Users:
1. [Main README](../README.md) - Get started
2. [Pipeline Explained](ENSEMBLE_PIPELINE_EXPLAINED.md) - Understand the approach
3. [Guides](GUIDES.md) - Follow setup instructions
4. [Scripts Summary](SCRIPTS_SUMMARY.md) - Run benchmarks

### For Researchers:
1. [Protocol](PROTOCOL.md) - Technical specification
2. [Alignment Analysis](ALIGNMENT_ANALYSIS.md) - Implementation details & fixes
3. [Field Names Comparison](FIELD_NAMES_COMPARISON.md) - Dataset compatibility
4. [benchmarks/README.md](../benchmarks/README.md) - Available evaluations

### For Contributors:
1. [Contributing](CONTRIBUTING.md) - Contribution guidelines
2. [Changelog](CHANGELOG.md) - Recent changes
3. [Alignment Analysis](ALIGNMENT_ANALYSIS.md) - Known issues and fixes

---

**Questions?** See the main [README](../README.md) or open an issue on GitHub.
