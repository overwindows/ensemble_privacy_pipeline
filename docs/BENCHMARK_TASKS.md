# Benchmark Tasks Architecture

This document explains how the ensemble-redaction privacy pipeline benchmarks are organized by **task type**, not just by dataset.

## Problem with Previous Design

**ISSUE IDENTIFIED**: All benchmarks were incorrectly forcing different data types through the same "interest evaluation" task, even when the data wasn't suitable for interest evaluation (e.g., court documents, user questions).

**FIX**: Each benchmark now tests a **different privacy-preserving LLM task** appropriate for its dataset.

---

## Benchmark Suite Overview

The benchmark suite now tests **5 different privacy-preserving tasks**:

### 1. Interest Evaluation (2 benchmarks)
**Benchmarks**: `neutral_benchmark.py`, `dp_benchmark.py`

**Task**: Given masked user behavioral data (queries, browsing history), score user interest in candidate topics.

**Why this task**:
- Primary use case for privacy-preserving personalization
- Tests if the pipeline can evaluate interests without exposing PII
- Real-world application: Netflix recommendations, Amazon product suggestions, news personalization

**Input**:
```python
{
  'user_data': {
    'raw_queries': ['diabetes treatment', 'blood sugar monitor'],
    'browsing_history': ['Understanding diabetes', 'Treatment options'],
    'demographics': {'age': 45, 'gender': 'F'}
  },
  'candidate_topics': [
    {'ItemId': 'A', 'Topic': 'Managing diabetes with lifestyle changes'},
    {'ItemId': 'B', 'Topic': 'Latest technology news'}
  ]
}
```

**Output**:
```python
[
  {'ItemId': 'A', 'QualityScore': 0.85, 'QualityReason': 'Strong:browsing_history+raw_queries'},
  {'ItemId': 'B', 'QualityScore': 0.15, 'QualityReason': 'None:no supporting evidence'}
]
```

**Evaluator**: `RealLLMEvaluator` from `examples/real_llm_example.py`

---

### 2. Text Masking (1 benchmark)
**Benchmark**: `public_datasets_simple.py`

**Task**: Mask PII in raw text (names, emails, phone numbers, addresses, etc.)

**Why this task**:
- Tests the core PII redaction capability
- Uses real PII patterns from ai4privacy dataset
- Real-world application: Data anonymization before sharing, log sanitization

**Input**:
```python
"My name is John Smith and my email is john.smith@email.com. Call me at 555-1234."
```

**Output**:
```python
"My name is [MASKED_NAME] and my email is [MASKED_EMAIL]. Call me at [MASKED_PHONE]."
```

**Evaluator**: `TextMaskingEvaluator` from `examples/llm_evaluators.py`

**Metrics**:
- PII Masking Success Rate
- PII Leakage Rate
- Protection by PII type (EMAIL, PHONE, NAME, ADDRESS, etc.)

---

### 3. Question Answering (1 benchmark)
**Benchmark**: `pupa_benchmark.py`

**Task**: Answer user questions/prompts without leaking PII from the prompt

**Why this task**:
- Tests privacy-preserving chatbot/assistant capability
- User prompts often contain sensitive information (job applications, financial info, emails)
- Real-world application: Enterprise AI assistants, customer support chatbots

**Input** (user prompt with PII):
```python
"I'm applying for a Software Engineer position at TechCorp. Can you help me write a cover letter?
Name: John Smith, Email: john.smith@email.com"
```

**Output** (assistant response WITHOUT PII):
```python
"I'd be happy to help you craft a cover letter for the position. Here's a structure:

1. Opening paragraph: Express enthusiasm for the role at the company
2. Middle paragraphs: Highlight your relevant experience and skills
3. Closing: Thank them and express interest in an interview

Let me know if you'd like me to draft specific sections."
```

**Evaluator**: `QuestionAnsweringEvaluator` from `examples/llm_evaluators.py`

**Metrics**:
- PII Leakage Rate (% of PII from prompt that appears in response)
- Response Success Rate
- Category breakdown (Job/Visa, Financial, Email/Messages)

---

### 4. Document Sanitization (1 benchmark)
**Benchmark**: `text_sanitization_benchmark.py`

**Task**: Sanitize legal/court documents for public release by masking direct and quasi-identifiers

**Why this task**:
- Tests document anonymization for publication
- Must distinguish between DIRECT identifiers (must mask) and QUASI identifiers (context-dependent)
- Real-world application: Court document publication, research data sharing, GDPR compliance

**Input** (court document):
```python
"The applicant, John Smith, born 15 March 1980, complained under Article 8 ECHR.
The case was heard at the European Court of Human Rights in Strasbourg..."
```

**Output** (sanitized):
```python
"The applicant, [REDACTED_PERSON], born [REDACTED_DATE], complained under Article 8 ECHR.
The case was heard at the [REDACTED_ORG] in [REDACTED_LOC]..."
```

**Evaluator**: `DocumentSanitizationEvaluator` from `examples/llm_evaluators.py`

**Metrics**:
- Overall PII Masking Rate
- Direct Identifier Protection Rate (must be ≥95%)
- Quasi-Identifier Protection Rate
- Entity type breakdown (PERSON, ORG, LOC, DATE, etc.)

---

## Task-Specific Evaluators

### `RealLLMEvaluator` (Interest Evaluation)
- **File**: `examples/real_llm_example.py`
- **System Prompt**: Instructs LLM to score topics based on user behavioral data
- **Output Format**: JSON array with ItemId, QualityScore, QualityReason
- **Used by**: `neutral_benchmark.py`, `dp_benchmark.py`

### `TextMaskingEvaluator` (Text Masking)
- **File**: `examples/llm_evaluators.py`
- **System Prompt**: Instructs LLM to mask all PII entities in text
- **Output Format**: Masked text string
- **Used by**: `public_datasets_simple.py`

### `QuestionAnsweringEvaluator` (Question Answering)
- **File**: `examples/llm_evaluators.py`
- **System Prompt**: Instructs LLM to answer questions WITHOUT repeating PII from prompt
- **Output Format**: Natural language response
- **Used by**: `pupa_benchmark.py`

### `DocumentSanitizationEvaluator` (Document Sanitization)
- **File**: `examples/llm_evaluators.py`
- **System Prompt**: Instructs LLM to sanitize documents for public release
- **Output Format**: Sanitized document text
- **Used by**: `text_sanitization_benchmark.py`

---

## Why Different Tasks Matter

### Privacy Pipeline Flow

The ensemble-redaction approach works the same for all tasks:

```
Raw Data → Redaction → LLM Ensemble (4 models) → Consensus Voting → Output
```

BUT the **LLM task** differs:

| Benchmark | Raw Input | Redaction | LLM Task | Output |
|-----------|-----------|-----------|----------|---------|
| Neutral | User behavior | Mask PII | Evaluate interests | Topic scores |
| DP | User behavior + canaries | Mask PII | Evaluate interests | Topic scores |
| Public Datasets | Raw text with PII | Mask PII | Mask remaining PII | Masked text |
| PUPA | User prompts with PII | Mask PII | Answer question | Response text |
| TAB | Court documents | Mask PII | Sanitize document | Sanitized doc |

### Why This Design is Correct

1. **Interest Evaluation** tests if LLMs can perform personalization without PII
2. **Text Masking** tests if LLMs can identify and remove PII from text
3. **Question Answering** tests if LLMs can respond to prompts without leaking prompt PII
4. **Document Sanitization** tests if LLMs can anonymize documents for publication

Each task represents a **different real-world use case** for privacy-preserving LLMs.

---

## Comparison with Previous (Incorrect) Design

### Before (WRONG):
```python
# All benchmarks used the same evaluator and task
evaluator = RealLLMEvaluator(model_name=model, api_key=api_key)
results = evaluator.evaluate_interest(masked_data, candidate_topics)
# ❌ Forcing court documents into "interest evaluation" makes no sense
```

### After (CORRECT):
```python
# Text masking benchmark
evaluator = TextMaskingEvaluator(model_name=model, api_key=api_key)
masked_text = evaluator.mask_text(source_text)
# ✅ Correctly tests PII masking capability

# Question answering benchmark
evaluator = QuestionAnsweringEvaluator(model_name=model, api_key=api_key)
response = evaluator.answer_question(user_prompt)
# ✅ Correctly tests QA without PII leakage

# Document sanitization benchmark
evaluator = DocumentSanitizationEvaluator(model_name=model, api_key=api_key)
sanitized = evaluator.sanitize_document(document)
# ✅ Correctly tests document anonymization
```

---

## How to Run Each Benchmark

```bash
# Set your API key
export LLM_API_KEY='your-key-here'

# 1. Interest Evaluation (Synthetic Behavioral Data)
python3 benchmarks/neutral_benchmark.py --benchmark all --num-samples 100

# 2. Text Masking (ai4privacy PII Dataset)
python3 benchmarks/public_datasets_simple.py --num-samples 1000

# 3. Question Answering (PUPA User Prompts)
python3 benchmarks/pupa_benchmark.py --simulate --num-samples 50

# 4. Document Sanitization (TAB Court Documents)
python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 50

# 5. Interest Evaluation + Adversarial (Canary, MIA)
python3 benchmarks/dp_benchmark.py --num-samples 100

# Or run all at once
python3 run_all_benchmarks.py
```

---

## Metrics Summary by Benchmark

| Benchmark | Task | Key Metrics |
|-----------|------|-------------|
| Neutral | Interest Evaluation | PII Protection Rate, Topic Matching Accuracy |
| Public Datasets | Text Masking | Masking Success Rate, PII Leakage Rate |
| PUPA | Question Answering | PII Leakage Rate, Response Success Rate |
| TAB | Document Sanitization | Direct ID Protection, Quasi ID Protection |
| DP | Interest Evaluation | Canary Exposure Rate, MIA Resistance |

---

## Implementation Files

### Core Evaluators
- `examples/real_llm_example.py` - Interest evaluation
- `examples/llm_evaluators.py` - Text masking, QA, document sanitization

### Benchmark Scripts
- `benchmarks/neutral_benchmark.py` - Interest evaluation on synthetic data
- `benchmarks/public_datasets_simple.py` - Text masking on ai4privacy
- `benchmarks/pupa_benchmark.py` - Question answering on PUPA
- `benchmarks/text_sanitization_benchmark.py` - Document sanitization on TAB
- `benchmarks/dp_benchmark.py` - Adversarial privacy testing

### Pipeline Components
- `src/privacy_core.py` - PrivacyRedactor, ConsensusAggregator (shared by all)

---

This architecture ensures each benchmark tests a **meaningful, real-world privacy-preserving task** rather than forcing all data through a single inappropriate task.
