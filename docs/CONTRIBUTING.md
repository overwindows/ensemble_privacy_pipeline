# Contributing to Ensemble-Redaction Pipeline

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## 🎯 Areas for Contribution

We welcome contributions in the following areas:

### 1. LLM Provider Support
- [ ] Google Gemini integration
- [ ] Mistral AI integration
- [ ] Cohere integration
- [ ] Open-source models via vLLM/Ollama
- [ ] Azure OpenAI Service

### 2. Privacy Enhancements
- [ ] Optional formal DP noise layer
- [ ] TEE integration (AWS Nitro, Intel SGX, Azure Confidential VMs)
- [ ] Additional masking strategies
- [ ] Privacy metrics improvements
- [ ] Reconstruction attack simulations

### 3. Consensus Methods
- [ ] Weighted voting (trust certain models more)
- [ ] Bayesian consensus
- [ ] Evidence fusion algorithms
- [ ] Adaptive thresholding

### 4. Performance Improvements
- [ ] Async/parallel processing optimizations
- [ ] Caching strategies
- [ ] Batch processing
- [ ] Rate limiting and retry logic

### 5. Testing & Validation
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Privacy metric tests
- [ ] Performance benchmarks
- [ ] Adversarial testing

### 6. Documentation
- [ ] API reference
- [ ] Deployment guides (AWS, Azure, GCP)
- [ ] Best practices guide
- [ ] Case studies
- [ ] Tutorial videos

---

## 🚀 Getting Started

### 1. Fork the Repository

```bash
# Click "Fork" on GitHub, then clone your fork
git clone https://github.com/yourusername/ensemble-privacy-pipeline.git
cd ensemble-privacy-pipeline
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## 📝 Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for all functions
- Maximum line length: **88 characters** (Black default)
- Use **descriptive variable names**

### Code Formatting

```bash
# Format code with Black
black .

# Check style with flake8
flake8 .

# Type check with mypy
mypy .
```

### Example

```python
def aggregate_scores(
    model_results: List[List[Dict[str, Any]]],
    method: str = "median"
) -> List[Dict[str, Any]]:
    """
    Aggregate scores from multiple models.

    Args:
        model_results: List of results from each model
        method: Aggregation method ("median", "mean", "trimmed_mean")

    Returns:
        Consensus results

    Raises:
        ValueError: If method is not supported
    """
    # Implementation
    pass
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_redactor.py
```

### Writing Tests

```python
# tests/test_redactor.py
import pytest
from ensemble_privacy_pipeline import PrivacyRedactor

def test_query_masking():
    """Test that queries are properly masked."""
    redactor = PrivacyRedactor()

    raw_data = {
        "BingSearch": [
            {"query": "diabetes treatment", "timestamp": "2024-01-15T10:30:00"}
        ]
    }

    masked = redactor.redact_user_data(raw_data)

    # Should not contain original query
    assert "diabetes" not in str(masked)
    # Should contain token
    assert "QUERY_SEARCH" in str(masked)

def test_navigation_noise_filtering():
    """Test that navigation queries are filtered."""
    redactor = PrivacyRedactor()

    assert redactor._is_navigation_noise("youtube.com") == True
    assert redactor._is_navigation_noise("login") == True
    assert redactor._is_navigation_noise("diabetes treatment") == False
```

### Privacy Tests

All changes must pass privacy validation:

```python
def test_no_pii_leakage(output):
    """Ensure no PII in output."""
    output_str = json.dumps(output)

    # Check for common PII patterns
    assert not re.search(r'\b\d{3}-\d{2}-\d{4}\b', output_str)  # SSN
    assert not re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', output_str)  # Email

    # Check for specific query leakage
    assert "diabetes" not in output_str.lower()
    assert "medication" not in output_str.lower()
```

---

## 📋 Pull Request Process

### 1. Before Submitting

- [ ] Code passes all tests (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] No style violations (`flake8 .`)
- [ ] Type checks pass (`mypy .`)
- [ ] Documentation updated
- [ ] Privacy tests pass
- [ ] Commit messages are descriptive

### 2. Commit Message Format

```
type(scope): brief description

Longer description if needed.

- Bullet points for details
- Include issue references (#123)

Closes #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:
```
feat(consensus): add weighted voting method

Implement weighted consensus that allows trusting certain models more.
Useful when some models have better track record on specific domains.

- Add weights parameter to ConsensusAggregator
- Update tests for weighted voting
- Add example in docs

Closes #42
```

### 3. Submit Pull Request

1. Push to your fork
2. Open PR on main repository
3. Fill out PR template
4. Wait for review

### 4. Review Process

- Maintainers will review within 3-5 days
- Address feedback
- Once approved, maintainer will merge

---

## 🐛 Reporting Bugs

### Security Issues

**DO NOT** open public issues for security vulnerabilities.

Email: security@example.com (use PGP key if available)

### Bug Reports

Use GitHub Issues with template:

```markdown
**Describe the bug**
Clear description of the issue

**To Reproduce**
Steps to reproduce:
1. Load data '...'
2. Run pipeline with '...'
3. See error

**Expected behavior**
What you expected to happen

**Actual behavior**
What actually happened

**Environment**
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.9.7]
- Package version: [e.g. 1.0.0]

**Additional context**
Any other relevant information
```

---

## 💡 Feature Requests

Use GitHub Issues with template:

```markdown
**Feature description**
Clear description of the feature

**Use case**
Why is this needed? What problem does it solve?

**Proposed solution**
How should it work?

**Alternatives considered**
Other approaches you've thought about

**Additional context**
Any other relevant information
```

---

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Prioritize user privacy and security
- Maintain professional conduct

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Unethical use of the code (e.g., for surveillance)

### Enforcement

Violations will result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to: conduct@example.com

---

## 🎓 Learning Resources

### Privacy & Security
- [Differential Privacy Book](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [PATE Paper](https://arxiv.org/abs/1610.05755)
- [GDPR Guidelines](https://gdpr.eu/)

### Python Development
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Black Formatter](https://black.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)

---

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in paper citations (if applicable)

---

## 📞 Questions?

- GitHub Discussions: For general questions
- GitHub Issues: For bugs and features
- Email: contribute@example.com

---

Thank you for making privacy-preserving AI better! 🛡️
