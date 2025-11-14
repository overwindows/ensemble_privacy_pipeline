# Alignment Analysis: Implementation vs Real Datasets

## ⚠️ CRITICAL ISSUE IDENTIFIED

The `PrivacyRedactor` in `src/privacy_core.py` is **HARDCODED for Microsoft-specific fields** and does NOT work with real public datasets!

## The Problem

### What PrivacyRedactor Actually Supports (Lines 85-164)

```python
def redact_user_data(self, raw_user_data: Dict) -> Dict:
    # ONLY handles these specific fields:
    if "MSNClicks" in raw_user_data:        # ❌ Microsoft-specific
    if "BingSearch" in raw_user_data:       # ❌ Microsoft-specific
    if "BingClickedQueries" in raw_user_data:  # ❌ Microsoft-specific
    if "MSNUpvotes" in raw_user_data:       # ❌ Microsoft-specific
    if "MAI" in raw_user_data:              # ❌ Microsoft-specific
    if "demographics" in raw_user_data:     # ✅ Generic (OK)
```

**Missing support for:**
- ❌ `raw_queries` (used in neutral_benchmark.py)
- ❌ `browsing_history` (used in neutral_benchmark.py)
- ❌ `source_text` (ai4privacy dataset)
- ❌ `user_prompt` (PUPA dataset)
- ❌ `text` (TAB dataset)

## Dataset Format Mismatches

### 1. ai4privacy/pii-masking-200k (public_datasets_simple.py)

**Dataset Format:**
```python
{
    'source_text': "My name is Sarah Johnson...",
    'masked_text': "[NAME] and I live at..."
}
```

**What Our Pipeline Does:**
```python
# benchmarks/public_datasets_simple.py line 140
user_data = {
    'raw_queries': [sample['source_text']],  # ❌ Not handled by PrivacyRedactor!
    'demographics': {}
}
masked_data = redactor.redact_user_data(user_data)  # Returns empty dict!
```

**Result:** ❌ PrivacyRedactor ignores `raw_queries`, returns nearly empty dict

---

### 2. PUPA (pupa_benchmark.py)

**Dataset Format:**
```python
{
    'user_prompt': "I'm applying for a Software Engineer position...",
    'pii_units': ['John Smith', 'TechCorp Inc']
}
```

**What Our Pipeline Does:**
```python
# benchmarks/pupa_benchmark.py line 140
user_data = {
    'raw_queries': [sample['user_prompt']],  # ❌ Not handled by PrivacyRedactor!
    'demographics': {}
}
```

**Result:** ❌ PrivacyRedactor ignores `raw_queries`, returns nearly empty dict

---

### 3. TAB (text_sanitization_benchmark.py)

**Dataset Format:**
```python
{
    'text': "The applicant, John Smith, complained...",
    'annotations': [...]
}
```

**What Our Pipeline Does:**
```python
# benchmarks/text_sanitization_benchmark.py line 140
user_data = {
    'raw_queries': [sample['text'][:500]],  # ❌ Not handled by PrivacyRedactor!
    'demographics': {}
}
```

**Result:** ❌ PrivacyRedactor ignores `raw_queries`, returns nearly empty dict

---

### 4. Vendor-Neutral Benchmark (neutral_benchmark.py)

**Data Format:**
```python
user_data = {
    'raw_queries': ['diabetes symptoms', ...],       # ❌ Not handled!
    'browsing_history': ['Article title', ...],      # ❌ Not handled!
    'demographics': {'age': 42}                      # ✅ Handled
}
```

**Result:** ❌ Only demographics are redacted, queries/browsing ignored

---

### 5. Old Microsoft Format (run_benchmarks.py)

**Data Format:**
```python
user_data = {
    'MSNClicks': [...],   # ✅ Handled
    'BingSearch': [...],  # ✅ Handled
    'MAI': [...],         # ✅ Handled
    'demographics': {...} # ✅ Handled
}
```

**Result:** ✅ **ONLY THIS FORMAT WORKS!**

## Impact on Evaluation

### Working Benchmarks ✅
1. **`run_benchmarks.py`** - Uses MSNClicks/BingSearch
2. **`run_demo_pipeline.py`** - Uses MSNClicks/BingSearch

### Broken Benchmarks ❌
1. **`benchmarks/public_datasets_simple.py`** - Uses `raw_queries` → **NOT REDACTED**
2. **`benchmarks/pupa_benchmark.py`** - Uses `raw_queries` → **NOT REDACTED**
3. **`benchmarks/text_sanitization_benchmark.py`** - Uses `raw_queries` → **NOT REDACTED**
4. **`benchmarks/neutral_benchmark.py`** - Uses `raw_queries`/`browsing_history` → **NOT REDACTED**
5. **`benchmarks/dp_benchmark.py`** - Uses `raw_queries` → **NOT REDACTED**

## Why This Happened

The PrivacyRedactor was built for Microsoft's internal use case and never generalized for:
1. Public dataset formats
2. Vendor-neutral field names
3. Generic text inputs

## The Fix Required

### Option 1: Update PrivacyRedactor to Handle All Formats (Recommended)

Add support for generic field names in `src/privacy_core.py`:

```python
def redact_user_data(self, raw_user_data: Dict) -> Dict:
    masked = {}

    # GENERIC: raw_queries (NEW)
    if "raw_queries" in raw_user_data:
        masked_queries = []
        for query in raw_user_data["raw_queries"]:
            masked_token = self._mask_query(query, category="QUERY")
            if masked_token:
                masked_queries.append({"token": masked_token})
        if masked_queries:
            masked["queries"] = masked_queries

    # GENERIC: browsing_history (NEW)
    if "browsing_history" in raw_user_data:
        masked_browsing = []
        for item in raw_user_data["browsing_history"]:
            masked_token = self._mask_query(item, category="BROWSING")
            if masked_token:
                masked_browsing.append({"token": masked_token})
        if masked_browsing:
            masked["browsing"] = masked_browsing

    # Keep existing Microsoft-specific handlers for backward compatibility
    if "MSNClicks" in raw_user_data:
        # ... existing code ...

    if "BingSearch" in raw_user_data:
        # ... existing code ...
```

### Option 2: Adapter Layer

Create format adapters that convert public dataset formats to Microsoft format before redaction.

### Option 3: Separate Redactors

Create a `GenericRedactor` class for public datasets, keep `PrivacyRedactor` for Microsoft format.

## Testing Verification

To verify if benchmarks actually work:

```bash
# This will FAIL - redactor returns empty dict for generic fields
python3 benchmarks/neutral_benchmark.py --benchmark privacy_leakage --num-samples 2

# This will WORK - uses Microsoft fields
python3 run_benchmarks.py --benchmark privacy_leakage --num-samples 2
```

## Recommendation

**URGENT: Fix PrivacyRedactor to support generic field names** before claiming the benchmarks work with real datasets.

Current state:
- ✅ Code runs without errors
- ❌ But privacy redaction is **NOT ACTUALLY HAPPENING** for public datasets!
- ❌ Benchmarks are measuring "no redaction" not "ensemble redaction"

The ensemble consensus might still filter some PII through model behavior, but the input redaction step (which is claimed as a key privacy mechanism) is **completely bypassed** for all public dataset benchmarks.

## Summary Table

| Script | Field Names Used | PrivacyRedactor Support | Actually Redacts? |
|--------|------------------|------------------------|-------------------|
| `run_benchmarks.py` | MSNClicks, BingSearch | ✅ Yes | ✅ Yes |
| `run_demo_pipeline.py` | MSNClicks, BingSearch | ✅ Yes | ✅ Yes |
| `public_datasets_simple.py` | raw_queries | ❌ No | ❌ **NO** |
| `pupa_benchmark.py` | raw_queries | ❌ No | ❌ **NO** |
| `text_sanitization_benchmark.py` | raw_queries | ❌ No | ❌ **NO** |
| `neutral_benchmark.py` | raw_queries, browsing_history | ❌ No | ❌ **NO** |
| `dp_benchmark.py` | raw_queries | ❌ No | ❌ **NO** |

## Action Taken

1. ✅ Identified the misalignment issue
2. ✅ **FIXED: Updated `src/privacy_core.py` to support generic field names**
   - Added support for `raw_queries` (list of queries/prompts)
   - Added support for `browsing_history` (list of browsing items)
   - Added support for `source_text` (single text - ai4privacy dataset)
   - Added support for `user_prompt` (single prompt - PUPA dataset)
   - Added support for `text` (generic text - TAB dataset)
   - Maintained backward compatibility with Microsoft-specific fields
3. ⏭️ Next: Verify all benchmarks actually redact data
4. ⏭️ Next: Re-test evaluation metrics after fix

## Fix Details

### Changes to `src/privacy_core.py` (Lines 98-156)

Added vendor-neutral field handlers before Microsoft-specific handlers:

```python
# GENERIC: raw_queries - List of search queries or user prompts
if "raw_queries" in raw_user_data:
    masked_queries = []
    for query in raw_user_data["raw_queries"]:
        if isinstance(query, str):
            masked_token = self._mask_query(query, category="QUERY")
            if masked_token:
                masked_queries.append({"token": masked_token})
    if masked_queries:
        masked["queries"] = masked_queries

# GENERIC: browsing_history - List of web pages/articles viewed
if "browsing_history" in raw_user_data:
    # ... similar pattern

# GENERIC: source_text, user_prompt, text - Single text fields
# ... handlers for each dataset format
```

### What This Fixes

All 7 benchmark scripts now properly redact data:

| Benchmark | Field Names | Redacts? | Status |
|-----------|-------------|----------|--------|
| run_benchmarks.py | MSNClicks, BingSearch | ✅ Yes | Working |
| run_demo_pipeline.py | MSNClicks, BingSearch | ✅ Yes | Working |
| public_datasets_simple.py | raw_queries | ✅ **NOW WORKS** | **FIXED** |
| pupa_benchmark.py | raw_queries | ✅ **NOW WORKS** | **FIXED** |
| text_sanitization_benchmark.py | raw_queries | ✅ **NOW WORKS** | **FIXED** |
| neutral_benchmark.py | raw_queries, browsing_history | ✅ **NOW WORKS** | **FIXED** |
| dp_benchmark.py | raw_queries | ✅ **NOW WORKS** | **FIXED** |
