# Field Names Comparison: Vendor-Specific vs Vendor-Neutral

## Problem Statement

The original `run_benchmarks.py` uses **Microsoft-specific field names** that do NOT align with real public datasets. This creates confusion and makes the code less reusable for general research.

## Comparison Table

| Aspect | Microsoft-Specific (`run_benchmarks.py`) | Vendor-Neutral (`neutral_benchmark.py`) | Real Public Datasets |
|--------|------------------------------------------|----------------------------------------|---------------------|
| **Search Queries** | `BingSearch` | `raw_queries` | `source_text`, `user_prompt`, `query` |
| **Browsing History** | `MSNClicks` | `browsing_history` | `text`, `masked_text` |
| **Interest Signals** | `MAI` (Microsoft Audience Intelligence) | Not needed | Not present |
| **Demographics** | `demographics` | `demographics` | `demographics` (sometimes) |
| **Use Case** | Microsoft internal | General research | Public datasets |
| **Alignment** | ❌ Microsoft-only | ✅ Generic | ✅ Standard formats |

## Example Comparison

### Microsoft-Specific (OLD - Don't Use)
```python
user_data = {
    'MSNClicks': [  # ❌ Microsoft-specific
        {'title': 'Understanding diabetes symptoms', 'timestamp': '2024-01-15T10:00:00'},
    ],
    'BingSearch': [  # ❌ Microsoft-specific
        {'query': 'diabetes diet plan', 'timestamp': '2024-01-15T11:00:00'},
    ],
    'MAI': ['Health'] * 8,  # ❌ Microsoft internal telemetry
    'demographics': {'age': 42, 'gender': 'F'}
}
```

### Vendor-Neutral (NEW - Recommended)
```python
user_data = {
    'raw_queries': [  # ✅ Generic search queries
        'diabetes symptoms',
        'diabetes diet plan',
    ],
    'browsing_history': [  # ✅ Generic browsing
        'Understanding diabetes symptoms',
        'Treatment options for diabetes',
    ],
    'demographics': {'age': 42, 'gender': 'F'}
}
```

### Real Public Datasets Format

**PUPA (NAACL 2025)**:
```python
{
    'user_prompt': "I'm applying for a Software Engineer position...",
    'pii_units': ['John Smith', 'TechCorp Inc', 'john@email.com']
}
```

**TAB (Text Anonymization Benchmark)**:
```python
{
    'text': "The applicant, John Smith, complained about violations...",
    'annotations': [
        {'span_text': 'John Smith', 'entity_type': 'PERSON', 'identifier_type': 'DIRECT'}
    ]
}
```

**ai4privacy/pii-masking-200k**:
```python
{
    'source_text': "My name is Sarah Johnson and I live at 123 Oak Street...",
    'masked_text': "[NAME] and I live at [ADDRESS]..."
}
```

## Why This Matters

### 1. **Research Reproducibility**
Vendor-specific names make it hard for other researchers to understand and reproduce your work.

### 2. **Dataset Alignment**
Your synthetic benchmarks should match the structure of real public datasets you'll evaluate on.

### 3. **Code Reusability**
Generic field names make the code reusable across different LLM providers and use cases.

### 4. **Transparency**
Clear, vendor-neutral names make it obvious this is general privacy research, not product-specific code.

## Recommended Scripts

### ✅ Use These (Vendor-Neutral)
1. **`benchmarks/neutral_benchmark.py`** ⭐ NEW
   - Generic field names: `raw_queries`, `browsing_history`
   - Works for any LLM privacy pipeline
   - Aligns with research datasets

2. **`benchmarks/public_datasets_simple.py`** ⭐
   - Uses real ai4privacy dataset
   - Loads actual PII data from Hugging Face

3. **`benchmarks/pupa_benchmark.py`** ⭐
   - Uses PUPA (NAACL 2025) format
   - Real user prompts with PII

4. **`benchmarks/text_sanitization_benchmark.py`** ⭐
   - Uses TAB (Text Anonymization Benchmark)
   - Real ECHR court cases

5. **`benchmarks/dp_benchmark.py`** ⭐
   - Canary exposure & MIA tests
   - Generic synthetic data

### ⚠️ Avoid These (Microsoft-Specific)
1. **`run_benchmarks.py`**
   - Uses `MSNClicks`, `BingSearch`, `MAI`
   - Microsoft-specific schema
   - Does NOT align with public datasets
   - **Keep for legacy compatibility only**

## Migration Guide

If you have code using the old Microsoft-specific format:

```python
# OLD (Microsoft-specific)
user_data = {
    'MSNClicks': [...],
    'BingSearch': [...],
    'MAI': [...]
}

# NEW (Vendor-neutral)
user_data = {
    'raw_queries': [query['query'] for query in old_data['BingSearch']],
    'browsing_history': [click['title'] for click in old_data['MSNClicks']],
    # MAI is removed - not needed for generic benchmarks
}
```

## Updated Benchmark Commands

### Recommended (Vendor-Neutral)
```bash
# Vendor-neutral synthetic benchmarks
python3 benchmarks/neutral_benchmark.py --benchmark all --num-samples 20

# Real public datasets
python3 benchmarks/public_datasets_simple.py --num-samples 100
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 50
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 50
python3 benchmarks/dp_benchmark.py --num-samples 20
```

### Legacy (Keep for Compatibility)
```bash
# Microsoft-specific synthetic benchmarks (legacy)
python3 run_benchmarks.py --benchmark all --num-samples 20
```

## Conclusion

**Use vendor-neutral field names** in all new code and benchmarks. This makes your research:
- ✅ More reproducible
- ✅ Aligned with public datasets
- ✅ Reusable across different LLM providers
- ✅ Clearer for the research community

The old `run_benchmarks.py` is kept for backward compatibility but should not be used for new work.
