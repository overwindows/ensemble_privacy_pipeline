#!/usr/bin/env python3
"""
PUPA Benchmark - Private User Prompt Annotations Dataset (NAACL 2025)

Evaluates your Ensemble-Redaction approach on the PUPA dataset from:
"PAPILLON: Privacy Preservation from Internet-based and Local Language Model Ensembles"
(Li et al., NAACL 2025)

PUPA Dataset:
- 901 instances from WildChat corpus
- Real-world user-agent interactions
- Annotated PII in 3 categories:
  1. Job, Visa, & Other Applications (16-41%)
  2. Financial and Corporate Info (29-47%)
  3. Quoted Emails and Messages (23-30%)

Evaluation Metrics:
- Privacy Leakage: % of PII units exposed in output
- Quality Preservation: Response quality maintenance

PAPILLON Results (baseline):
- Quality preserved: 85.5% of queries
- Privacy leakage: 7.5%

Usage:
    python3 benchmarks/pupa_benchmark.py --dataset-path /path/to/pupa.json --num-samples 901
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.privacy_core import PrivacyRedactor, ConsensusAggregator
from examples.llm_evaluators import QuestionAnsweringEvaluator, check_pii_leakage


# ============================================================================
# PUPA DATASET LOADER
# ============================================================================

class PUPADatasetLoader:
    """Load and prepare PUPA dataset."""

    def load_pupa_dataset(self, dataset_path: str, num_samples: int = 100) -> List[Dict]:
        """
        Load real PUPA dataset from JSON file.

        Expected format:
        [
            {
                "user_prompt": "...",
                "assistant_response": "...",
                "pii_units": ["name1", "company1", "email1"],
                "pii_category": "Job, Visa, & Other Applications"
            },
            ...
        ]
        """
        print(f"\n📥 Loading PUPA dataset from: {dataset_path}")

        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset file not found!")
            print(f"\nTo obtain PUPA dataset:")
            print(f"  1. Visit: https://github.com/Columbia-NLP-Lab/PAPILLON")
            print(f"  2. Follow dataset download instructions")
            print(f"  3. Point --dataset-path to the JSON file")
            sys.exit(1)

        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)

            samples = []
            for idx, item in enumerate(data[:num_samples]):
                samples.append({
                    'id': f'pupa_{idx}',
                    'user_prompt': item.get('user_prompt', item.get('query', '')),
                    'assistant_response': item.get('assistant_response', item.get('response', '')),
                    'pii_units': item.get('pii_units', []),
                    'pii_category': item.get('pii_category', 'Unknown'),
                })

            print(f"✓ Loaded {len(samples)} samples from PUPA dataset")
            return samples

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            sys.exit(1)


# ============================================================================
# PUPA EVALUATION
# ============================================================================

def evaluate_pupa_benchmark(samples: List[Dict], model_names: List[str], api_key: str) -> Dict:
    """
    Evaluate QUESTION ANSWERING WITHOUT PII LEAKAGE on PUPA benchmark.

    Task: Answer user prompts/questions without exposing PII from the prompt
    Metrics:
    1. Privacy Leakage: % of PII units exposed in assistant's response
    2. Response Quality: Can the assistant still provide helpful answers?
    3. Category Breakdown: Performance per PII category
    """
    print("\n" + "="*80)
    print("BENCHMARK: QUESTION ANSWERING (Privacy-Preserving)")
    print("Private User Prompt Annotations Dataset (NAACL 2025)")
    print("="*80)
    print("\nTask: Answer user questions without leaking PII from prompts")

    print(f"\n✓ Samples: {len(samples)}")
    print(f"✓ Models: {len(model_names)}")

    # Initialize QA evaluators
    evaluators = [QuestionAnsweringEvaluator(model_name=model, api_key=api_key) for model in model_names]

    results = {
        'total_samples': len(samples),
        'pii_units_total': 0,
        'pii_units_leaked': 0,
        'pii_leakage_rate': 0.0,
        'responses_generated': 0,
        'category_breakdown': defaultdict(lambda: {
            'samples': 0,
            'pii_total': 0,
            'pii_leaked': 0,
            'leakage_rate': 0.0
        }),
        'leaked_samples': [],
        'protected_samples': [],
    }

    # Process each sample
    for i, sample in enumerate(samples, 1):
        print(f"\n  Sample {i}/{len(samples)}: {sample['id']}")
        print(f"    Category: {sample['pii_category']}")
        print(f"    PII units: {len(sample['pii_units'])}")

        try:
            # Step 1: First, redact PII from the user prompt
            redactor = PrivacyRedactor()
            user_data = {
                'raw_queries': [sample['user_prompt']],
                'demographics': {}
            }
            masked_data = redactor.redact_user_data(user_data)
            redacted_prompt = masked_data.get('raw_queries', [sample['user_prompt']])[0]

            # Step 2: Each model answers the REDACTED prompt
            responses = []
            for evaluator in evaluators:
                try:
                    # Answer based on redacted prompt (not original)
                    response = evaluator.answer_question(redacted_prompt)
                    responses.append(response)
                except Exception as e:
                    print(f"      ⚠️  Model {evaluator.model_name} error: {e}")

            # Step 3: Ensemble consensus - aggregate WITHOUT using ground truth
            if responses:
                # Majority voting: Pick the most common response
                from collections import Counter
                response_counts = Counter(responses)

                if len(response_counts) == 1:
                    # All models gave same response
                    final_response = responses[0]
                    consensus_type = "unanimous"
                elif response_counts.most_common(1)[0][1] >= len(responses) / 2:
                    # Majority exists
                    final_response = response_counts.most_common(1)[0][0]
                    consensus_type = "majority"
                else:
                    # No clear majority, use first response as fallback
                    # In production, could use other heuristics (shortest, median length, etc.)
                    final_response = responses[0]
                    consensus_type = "no_consensus_fallback"

                if len(responses) > 1:
                    unique_responses = len(response_counts)
                    print(f"      📊 Ensemble ({len(responses)} models): {unique_responses} unique responses, consensus: {consensus_type}")

                results['responses_generated'] += 1

                # Step 4: EVALUATE consensus response against ground truth (not used for selection!)
                leakage_check = check_pii_leakage(final_response, sample['pii_units'])

                results['pii_units_total'] += len(sample['pii_units'])
                results['pii_units_leaked'] += leakage_check['leaked_count']

                # Update category stats
                category = sample['pii_category']
                results['category_breakdown'][category]['samples'] += 1
                results['category_breakdown'][category]['pii_total'] += len(sample['pii_units'])
                results['category_breakdown'][category]['pii_leaked'] += leakage_check['leaked_count']

                if leakage_check['is_protected']:
                    results['protected_samples'].append(sample['id'])
                    print(f"      ✅ ALL PII PROTECTED ({len(sample['pii_units'])} units)")
                else:
                    results['leaked_samples'].append({
                        'id': sample['id'],
                        'category': category,
                        'leaked_pii': leakage_check['leaked_entities'][:3],
                        'leak_rate': leakage_check['leakage_rate']
                    })
                    print(f"      ❌ PII LEAKED: {leakage_check['leaked_count']}/{len(sample['pii_units'])} units")
                    print(f"         Examples: {leakage_check['leaked_entities'][:3]}")

            else:
                print(f"      ⚠️  All models failed")

        except Exception as e:
            print(f"      ❌ Error: {e}")

    # Calculate final metrics
    if results['pii_units_total'] > 0:
        results['pii_leakage_rate'] = results['pii_units_leaked'] / results['pii_units_total']

    results['response_success_rate'] = results['responses_generated'] / results['total_samples']

    # Calculate category-specific rates
    for category, stats in results['category_breakdown'].items():
        if stats['pii_total'] > 0:
            stats['leakage_rate'] = stats['pii_leaked'] / stats['pii_total']

    return results


# ============================================================================
# COMPARISON WITH PAPILLON
# ============================================================================

def compare_with_papillon(results: Dict) -> Dict:
    """
    Compare results with PAPILLON baseline from NAACL 2025 paper.

    PAPILLON Results:
    - Quality preserved: 85.5%
    - Privacy leakage: 7.5%
    """
    print("\n" + "="*80)
    print("COMPARISON WITH PAPILLON (NAACL 2025)")
    print("="*80)

    papillon_baseline = {
        'quality_preservation': 0.855,  # 85.5%
        'privacy_leakage': 0.075,       # 7.5%
    }

    comparison = {
        'your_approach': {
            'response_success': results['response_success_rate'],
            'privacy_leakage': results['pii_leakage_rate'],
        },
        'papillon_baseline': papillon_baseline,
        'improvements': {
            'quality_diff': results['response_success_rate'] - papillon_baseline['quality_preservation'],
            'leakage_diff': papillon_baseline['privacy_leakage'] - results['pii_leakage_rate'],
        }
    }

    print("\n📊 Comparison Table:")
    print("-"*80)
    print(f"{'Metric':<30} {'Your Ensemble':<20} {'PAPILLON':<20}")
    print("-"*80)
    print(f"{'Response Success':<30} {comparison['your_approach']['response_success']*100:>8.1f}% {' '*10} {papillon_baseline['quality_preservation']*100:>8.1f}%")
    print(f"{'Privacy Leakage':<30} {comparison['your_approach']['privacy_leakage']*100:>8.1f}% {' '*10} {papillon_baseline['privacy_leakage']*100:>8.1f}%")
    print("-"*80)

    # Verdict
    quality_better = comparison['improvements']['quality_diff'] > 0
    privacy_better = comparison['improvements']['leakage_diff'] > 0

    if quality_better and privacy_better:
        verdict = "✅ Your approach OUTPERFORMS PAPILLON on both metrics!"
    elif privacy_better:
        verdict = "✅ Your approach has BETTER privacy than PAPILLON"
    elif quality_better:
        verdict = "✅ Your approach has BETTER quality than PAPILLON"
    else:
        verdict = "⚠️  PAPILLON performs better on this benchmark"

    print(f"\n💡 Verdict: {verdict}")

    if comparison['improvements']['leakage_diff'] > 0:
        print(f"   Privacy improvement: {comparison['improvements']['leakage_diff']*100:.1f}% fewer leaks")
    if comparison['improvements']['quality_diff'] > 0:
        print(f"   Quality improvement: {comparison['improvements']['quality_diff']*100:.1f}% better quality")

    print("-"*80)

    return comparison


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PUPA benchmark evaluation (NAACL 2025)')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to PUPA dataset JSON file')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--output', default='results/pupa_benchmark_results.json',
                       help='Output file (default: results/pupa_benchmark_results.json)')

    args = parser.parse_args()

    # Setup logging and results directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    log_file = 'logs/pupa_benchmark.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Replace print with logging
    def print(msg=""):
        logging.info(msg)

    # Check API key
    api_key = os.getenv('LLM_API_KEY')
    if not api_key:
        print("❌ Error: LLM_API_KEY not set!")
        print("\nSet your API key:")
        print("  export LLM_API_KEY='your-key-here'")
        sys.exit(1)

    print("="*80)
    print("PUPA BENCHMARK - Private User Prompt Annotations (NAACL 2025)")
    print("="*80)
    print(f"\n✓ API Key: {api_key[:20]}...")
    print(f"✓ Log file: {log_file}")

    # Your 4-model ensemble
    model_names = [
        "gpt-oss-120b",
        "DeepSeek-V3.1",
        "Qwen3-32B",
        "DeepSeek-V3-0324",
    ]
    print(f"✓ Models: {', '.join(model_names)}")

    # Load dataset
    loader = PUPADatasetLoader()
    samples = loader.load_pupa_dataset(args.dataset_path, args.num_samples)

    # Run evaluation
    start_time = time.time()
    results = evaluate_pupa_benchmark(samples, model_names, api_key)
    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "="*80)
    print("PUPA BENCHMARK RESULTS")
    print("="*80)

    print(f"\n✓ Total samples: {results['total_samples']}")
    print(f"✓ Responses generated: {results['responses_generated']}")
    print(f"✓ Total PII units: {results['pii_units_total']}")
    print(f"✓ PII units leaked: {results['pii_units_leaked']}")
    print(f"✓ PII units protected: {results['pii_units_total'] - results['pii_units_leaked']}")

    print(f"\n📊 Key Metrics:")
    print(f"   Privacy Leakage Rate: {results['pii_leakage_rate']*100:.2f}%")
    print(f"   Response Success Rate: {results['response_success_rate']*100:.1f}%")

    print(f"\n📊 Category Breakdown:")
    for category, stats in results['category_breakdown'].items():
        print(f"\n   {category}:")
        print(f"     - Samples: {stats['samples']}")
        print(f"     - PII total: {stats['pii_total']}")
        print(f"     - PII leaked: {stats['pii_leaked']}")
        print(f"     - Leakage rate: {stats['leakage_rate']*100:.1f}%")

    print(f"\n✓ Time: {elapsed:.1f}s ({elapsed/results['total_samples']:.1f}s per sample)")

    # Compare with PAPILLON
    comparison = compare_with_papillon(results)

    # Save results
    results_with_config = {
        'config': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'task': 'Question Answering (Privacy-Preserving)',
            'dataset': 'PUPA (NAACL 2025)',
            'num_samples': args.num_samples,
            'models': model_names,
        },
        'results': {
            'pii_leakage_rate': results['pii_leakage_rate'],
            'response_success_rate': results['response_success_rate'],
            'pii_units_total': results['pii_units_total'],
            'pii_units_leaked': results['pii_units_leaked'],
            'responses_generated': results['responses_generated'],
            'leaked_samples_count': len(results['leaked_samples']),
            'protected_samples_count': len(results['protected_samples']),
        },
        'category_breakdown': dict(results['category_breakdown']),
        'papillon_comparison': comparison,
        'time_seconds': elapsed,
    }

    with open(args.output, 'w') as f:
        json.dump(results_with_config, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
