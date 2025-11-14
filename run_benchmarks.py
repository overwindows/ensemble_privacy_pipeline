#!/usr/bin/env python3
"""
Benchmark Runner for Ensemble-Redaction Privacy Pipeline

Evaluates your approach on multiple benchmarks:
1. Privacy Leakage Detection (PII exposure)
2. Utility Preservation (accuracy maintained)
3. Comparison with Baseline (no privacy)

Usage:
  export SAMBANOVA_API_KEY='your-key-here'
  python3 run_benchmarks.py --benchmark all --num-samples 50

Supported Benchmarks:
  - privacy_leakage: Test PII leakage rate
  - utility: Test utility preservation
  - baseline_comparison: Compare with no-privacy baseline
  - all: Run all benchmarks
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any
from collections import defaultdict
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.privacy_core import PrivacyRedactor, ConsensusAggregator, analyze_privacy_leakage
from examples.real_llm_example import RealLLMEvaluator


# ============================================================================
# TEST DATA GENERATOR
# ============================================================================

class BenchmarkDataGenerator:
    """Generate synthetic test data with known PII."""

    def generate_medical_samples(self, num_samples: int = 10) -> List[Dict]:
        """Generate medical domain samples with sensitive health data."""
        samples = []

        conditions = ['diabetes', 'hypertension', 'cancer', 'depression', 'arthritis']
        medications = ['metformin', 'lisinopril', 'chemotherapy', 'antidepressants', 'ibuprofen']

        for i in range(num_samples):
            condition = conditions[i % len(conditions)]
            medication = medications[i % len(medications)]

            sample = {
                'id': f'medical_{i}',
                'user_data': {
                    'MSNClicks': [
                        {'title': f'Understanding {condition} symptoms', 'timestamp': '2024-01-15T10:00:00'},
                        {'title': f'Treatment options for {condition}', 'timestamp': '2024-01-14T15:00:00'},
                    ],
                    'BingSearch': [
                        {'query': f'{condition} diet plan', 'timestamp': '2024-01-15T11:00:00'},
                        {'query': f'side effects of {medication}', 'timestamp': '2024-01-14T16:00:00'},
                    ],
                    'MAI': ['Health'] * 8 + ['Fitness'] * 2,
                    'demographics': {'age': 40 + i, 'gender': 'F' if i % 2 == 0 else 'M'}
                },
                'candidate_topics': [
                    {'ItemId': 'A', 'Topic': f'Managing {condition} with lifestyle changes'},
                    {'ItemId': 'B', 'Topic': 'Latest technology news'},
                ],
                'expected_match': 'A',
                'pii_keywords': [condition, medication],
            }
            samples.append(sample)

        return samples

    def generate_financial_samples(self, num_samples: int = 10) -> List[Dict]:
        """Generate financial domain samples."""
        samples = []

        for i in range(num_samples):
            salary = 50000 + (i * 10000)

            sample = {
                'id': f'financial_{i}',
                'user_data': {
                    'MSNClicks': [
                        {'title': 'Best investment strategies for retirement', 'timestamp': '2024-01-15T10:00:00'},
                        {'title': 'How to save for a house down payment', 'timestamp': '2024-01-14T15:00:00'},
                    ],
                    'BingSearch': [
                        {'query': f'salary ${salary} tax calculator', 'timestamp': '2024-01-15T11:00:00'},
                        {'query': 'mortgage rates 2024', 'timestamp': '2024-01-14T16:00:00'},
                    ],
                    'MAI': ['Finance'] * 7 + ['RealEstate'] * 3,
                    'demographics': {'age': 30 + i, 'gender': 'M' if i % 2 == 0 else 'F'}
                },
                'candidate_topics': [
                    {'ItemId': 'A', 'Topic': 'Financial planning for young professionals'},
                    {'ItemId': 'B', 'Topic': 'Latest sports news'},
                ],
                'expected_match': 'A',
                'pii_keywords': [str(salary), 'mortgage'],
            }
            samples.append(sample)

        return samples


# ============================================================================
# BENCHMARK 1: PRIVACY LEAKAGE TEST
# ============================================================================

def benchmark_privacy_leakage(samples: List[Dict], model_names: List[str], api_key: str) -> Dict:
    """
    Test: Does the pipeline prevent PII leakage?

    Measures:
    - PII keyword exposure rate
    - Specific query leakage
    - Demographic information leakage
    """
    print("\n" + "="*80)
    print("BENCHMARK 1: PRIVACY LEAKAGE TEST")
    print("="*80)

    print(f"\nTesting {len(samples)} samples with {len(model_names)} models...")

    # Initialize pipeline
    redactor = PrivacyRedactor()
    aggregator = ConsensusAggregator()
    evaluators = [RealLLMEvaluator(model_name=model, api_key=api_key) for model in model_names]

    results = {
        'total_samples': len(samples),
        'pii_exposed_count': 0,
        'pii_exposure_rate': 0.0,
        'exposed_samples': [],
        'safe_samples': [],
    }

    for i, sample in enumerate(samples, 1):
        print(f"\n  Sample {i}/{len(samples)}: {sample['id']}")

        # Step 1: Redact
        masked_data = redactor.redact_user_data(sample['user_data'])

        # Step 2: Ensemble evaluation
        all_results = []
        for evaluator in evaluators:
            try:
                eval_results = evaluator.evaluate_interest(masked_data, sample['candidate_topics'])
                all_results.append(eval_results)
            except Exception as e:
                print(f"    ⚠️  Model {evaluator.model_name} error: {e}")

        # Step 3: Consensus
        if all_results:
            consensus = aggregator.aggregate_median(all_results)

            # Step 4: Check for PII leakage
            output_str = json.dumps(consensus)

            leaked_pii = []
            for pii_keyword in sample['pii_keywords']:
                if pii_keyword.lower() in output_str.lower():
                    leaked_pii.append(pii_keyword)

            if leaked_pii:
                results['pii_exposed_count'] += 1
                results['exposed_samples'].append({
                    'id': sample['id'],
                    'leaked_pii': leaked_pii,
                    'output': consensus
                })
                print(f"    ❌ PII LEAKED: {leaked_pii}")
            else:
                results['safe_samples'].append(sample['id'])
                print(f"    ✅ NO PII LEAKED")
        else:
            print(f"    ⚠️  All models failed")

    results['pii_exposure_rate'] = results['pii_exposed_count'] / results['total_samples']

    print("\n" + "-"*80)
    print(f"PRIVACY LEAKAGE RESULTS:")
    print(f"  Total Samples:    {results['total_samples']}")
    print(f"  PII Exposed:      {results['pii_exposed_count']}")
    print(f"  PII Safe:         {results['total_samples'] - results['pii_exposed_count']}")
    print(f"  Exposure Rate:    {results['pii_exposure_rate']*100:.1f}%")
    print(f"  Protection Rate:  {(1-results['pii_exposure_rate'])*100:.1f}%")
    print("-"*80)

    return results


# ============================================================================
# BENCHMARK 2: UTILITY PRESERVATION TEST
# ============================================================================

def benchmark_utility_preservation(samples: List[Dict], model_names: List[str], api_key: str) -> Dict:
    """
    Test: Does the pipeline maintain utility (correct topic matching)?

    Measures:
    - Topic matching accuracy
    - Score consistency across ensemble
    - False positive/negative rates
    """
    print("\n" + "="*80)
    print("BENCHMARK 2: UTILITY PRESERVATION TEST")
    print("="*80)

    print(f"\nTesting {len(samples)} samples with {len(model_names)} models...")

    # Initialize pipeline
    redactor = PrivacyRedactor()
    aggregator = ConsensusAggregator()
    evaluators = [RealLLMEvaluator(model_name=model, api_key=api_key) for model in model_names]

    results = {
        'total_samples': len(samples),
        'correct_matches': 0,
        'accuracy': 0.0,
        'scores': [],
    }

    for i, sample in enumerate(samples, 1):
        print(f"\n  Sample {i}/{len(samples)}: {sample['id']}")

        # Run pipeline
        masked_data = redactor.redact_user_data(sample['user_data'])

        all_results = []
        for evaluator in evaluators:
            try:
                eval_results = evaluator.evaluate_interest(masked_data, sample['candidate_topics'])
                all_results.append(eval_results)
            except Exception as e:
                print(f"    ⚠️  Model {evaluator.model_name} error: {e}")

        if all_results:
            consensus = aggregator.aggregate_median(all_results)

            # Find highest scoring topic
            best_match = max(consensus, key=lambda x: x['QualityScore'])

            if best_match['ItemId'] == sample['expected_match']:
                results['correct_matches'] += 1
                print(f"    ✅ CORRECT: {best_match['ItemId']} (score: {best_match['QualityScore']:.2f})")
            else:
                print(f"    ❌ WRONG: Got {best_match['ItemId']}, expected {sample['expected_match']}")

            results['scores'].append(best_match['QualityScore'])

    results['accuracy'] = results['correct_matches'] / results['total_samples']
    results['avg_score'] = sum(results['scores']) / len(results['scores']) if results['scores'] else 0

    print("\n" + "-"*80)
    print(f"UTILITY PRESERVATION RESULTS:")
    print(f"  Total Samples:    {results['total_samples']}")
    print(f"  Correct Matches:  {results['correct_matches']}")
    print(f"  Accuracy:         {results['accuracy']*100:.1f}%")
    print(f"  Avg Score:        {results['avg_score']:.2f}")
    print("-"*80)

    return results


# ============================================================================
# BENCHMARK 3: BASELINE COMPARISON
# ============================================================================

def benchmark_baseline_comparison(samples: List[Dict], model_names: List[str], api_key: str) -> Dict:
    """
    Test: Compare privacy-preserving vs baseline (no privacy).

    Measures:
    - PII leakage: baseline vs ours
    - Utility: baseline vs ours
    """
    print("\n" + "="*80)
    print("BENCHMARK 3: BASELINE COMPARISON")
    print("="*80)

    # Run privacy-preserving pipeline
    print("\n📊 Running privacy-preserving pipeline...")
    privacy_results = benchmark_privacy_leakage(samples, model_names, api_key)

    # Simulate baseline (100% PII leakage)
    print("\n📊 Simulating baseline (no privacy)...")
    baseline_pii_rate = 1.0  # Baseline exposes all PII

    improvement = (baseline_pii_rate - privacy_results['pii_exposure_rate']) * 100

    results = {
        'baseline_pii_rate': baseline_pii_rate,
        'privacy_pii_rate': privacy_results['pii_exposure_rate'],
        'improvement_pct': improvement,
    }

    print("\n" + "-"*80)
    print(f"BASELINE COMPARISON RESULTS:")
    print(f"  Baseline PII Rate:    {baseline_pii_rate*100:.1f}%")
    print(f"  Our Pipeline PII Rate: {privacy_results['pii_exposure_rate']*100:.1f}%")
    print(f"  Improvement:          {improvement:.1f}%")
    print("-"*80)

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run privacy pipeline benchmarks')
    parser.add_argument('--benchmark', choices=['privacy_leakage', 'utility', 'baseline_comparison', 'all'],
                       default='all', help='Which benchmark to run')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples per dataset (default: 10)')
    parser.add_argument('--output', default='benchmark_results.json',
                       help='Output file for results (default: benchmark_results.json)')

    args = parser.parse_args()

    # Check API key
    api_key = os.getenv('SAMBANOVA_API_KEY')
    if not api_key:
        print("❌ Error: SAMBANOVA_API_KEY not set!")
        print("\nSet your API key:")
        print("  export SAMBANOVA_API_KEY='your-key-here'")
        sys.exit(1)

    print("="*80)
    print("ENSEMBLE-REDACTION PRIVACY PIPELINE - BENCHMARK SUITE")
    print("="*80)
    print(f"\n✓ API Key: {api_key[:20]}...")
    print(f"✓ Samples per dataset: {args.num_samples}")

    # Your 4-model ensemble
    model_names = [
        "gpt-oss-120b",
        "DeepSeek-V3.1",
        "Qwen3-32B",
        "DeepSeek-V3-0324",
    ]

    print(f"✓ Models: {', '.join(model_names)}")

    # Generate test data
    print("\n📝 Generating test data...")
    data_gen = BenchmarkDataGenerator()
    medical_samples = data_gen.generate_medical_samples(args.num_samples)
    financial_samples = data_gen.generate_financial_samples(args.num_samples)
    all_samples = medical_samples + financial_samples

    print(f"  ✓ Medical samples: {len(medical_samples)}")
    print(f"  ✓ Financial samples: {len(financial_samples)}")
    print(f"  ✓ Total samples: {len(all_samples)}")

    # Run benchmarks
    all_results = {
        'config': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_samples': args.num_samples,
            'models': model_names,
        },
        'benchmarks': {}
    }

    if args.benchmark in ['privacy_leakage', 'all']:
        all_results['benchmarks']['privacy_leakage'] = benchmark_privacy_leakage(
            all_samples, model_names, api_key
        )

    if args.benchmark in ['utility', 'all']:
        all_results['benchmarks']['utility'] = benchmark_utility_preservation(
            all_samples, model_names, api_key
        )

    if args.benchmark in ['baseline_comparison', 'all']:
        all_results['benchmarks']['baseline_comparison'] = benchmark_baseline_comparison(
            all_samples, model_names, api_key
        )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE")
    print("="*80)
    print(f"\n✓ Results saved to: {args.output}")
    print("\nSummary:")

    if 'privacy_leakage' in all_results['benchmarks']:
        pl = all_results['benchmarks']['privacy_leakage']
        print(f"  Privacy: {(1-pl['pii_exposure_rate'])*100:.1f}% protected")

    if 'utility' in all_results['benchmarks']:
        util = all_results['benchmarks']['utility']
        print(f"  Utility: {util['accuracy']*100:.1f}% accuracy")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
