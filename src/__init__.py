"""
Ensemble-Redaction Consensus Pipeline - Core Components

This package contains the core privacy-preserving pipeline components.

For real LLM evaluation, use:
    from examples.real_llm_example import RealLLMEvaluator
"""

from .privacy_core import PrivacyRedactor, ConsensusAggregator, analyze_privacy_leakage

__all__ = ['PrivacyRedactor', 'ConsensusAggregator', 'analyze_privacy_leakage']
__version__ = '2.0.0'
