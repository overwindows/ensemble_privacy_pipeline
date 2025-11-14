"""
Ensemble-Redaction Consensus Pipeline - Core Components

This package contains the core privacy-preserving pipeline components.
"""

from .pipeline import PrivacyRedactor, MockLLMEvaluator, ConsensusAggregator

__all__ = ['PrivacyRedactor', 'MockLLMEvaluator', 'ConsensusAggregator']
__version__ = '1.0.0'
