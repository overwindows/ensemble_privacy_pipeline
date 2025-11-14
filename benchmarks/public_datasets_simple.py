#!/usr/bin/env python3
"""
Simplified Public Dataset Benchmark

Evaluates privacy pipeline on real public datasets:
- ai4privacy/pii-masking-200k (Hugging Face)

Usage:
    pip install datasets huggingface_hub
    python3 benchmarks/public_datasets_simple.py --num-samples 100
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any
from collections import defaultdict
import re

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.privacy_core import PrivacyRedactor, ConsensusAggregator, analyze_privacy_leakage
from examples.real_llm_example import RealLLMEvaluator


# ============================================================================
# DATASET LOADER
# ============================================================================

class DatasetLoader:
    """Load and prepare public privacy datasets."""

    def load_ai4privacy(self, num_samples: int = 100) -> List[Dict]:
        """
        Load ai4privacy/pii-masking-200k dataset from Hugging Face.

        This dataset contains:
        - 200K+ samples with PII
        - 54 different PII classes
        - Multiple languages (en, de, fr, it)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("❌ Error: 'datasets' package not installed")
            print("\nInstall with:")
            print("  pip install datasets huggingface_hub")
            sys.exit(1)

        print(f"\n📥 Loading ai4privacy/pii-masking-200k dataset...")
        print(f"   Samples requested: {num_samples}")

        try:
            # Load from Hugging Face
            dataset = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)

            samples = []
            for idx, sample in enumerate(dataset):
                if idx >= num_samples:
                    break

                # Extract fields
                source_text = sample.get('source_text', sample.get('text', ''))
                masked_text = sample.get('target_text', sample.get('masked_text', ''))

                if source_text:
                    # Extract PII entities
                    pii_entities = self._extract_pii_entities(source_text, masked_text)

                    samples.append({
                        'id': f'ai4privacy_{idx}',
                        'source_text': source_text,
                        'masked_text': masked_text,
                        'pii_entities': pii_entities,
                        'pii_count': len(pii_entities),
                    })

                if (idx + 1) % 10 == 0:
                    print(f"   Loaded {idx + 1}/{num_samples}...", end='\r')

            print(f"\n✓ Loaded {len(samples)} samples")
            return samples

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            sys.exit(1)

    def _extract_pii_entities(self, source: str, masked: str) -> List[str]:
        """Extract PII entities by comparing source and masked text."""
        # Find all [PII_TYPE] tags in masked text
        pii_pattern = r'\[([A-Z_]+)\]'
        pii_types = re.findall(pii_pattern, masked)
        return list(set(pii_types))


# ============================================================================
# BENCHMARK EVALUATOR
# ============================================================================

def evaluate_on_dataset(samples: List[Dict], model_names: List[str], api_key: str) -> Dict:
    """
    Evaluate privacy pipeline on real dataset samples.

    Tests:
    1. PII Leakage: Do any PII entities appear in output?
    2. Detection Rate: Can pipeline identify PII-containing samples?
    3. Utility: Is output still useful?
    """
    print("\n" + "="*80)
    print("BENCHMARK EVALUATION ON PUBLIC DATASET")
    print("="*80)

    # Initialize pipeline
    redactor = PrivacyRedactor()
    aggregator = ConsensusAggregator()
    evaluators = [RealLLMEvaluator(model_name=model, api_key=api_key) for model in model_names]

    results = {
        'total_samples': len(samples),
        'samples_with_pii': 0,
        'pii_leaked_count': 0,
        'pii_protected_count': 0,
        'pii_types_found': defaultdict(int),
        'failed_samples': [],
    }

    # Process each sample
    for i, sample in enumerate(samples, 1):
        print(f"\n  Sample {i}/{len(samples)}: {sample['id']}")
        print(f"    PII entities: {sample['pii_count']} ({', '.join(sample['pii_entities'][:3])}...)")

        if sample['pii_count'] > 0:
            results['samples_with_pii'] += 1

        try:
            # Convert to pipeline format
            user_data = {
                'raw_queries': [sample['source_text']],
                'demographics': {}
            }

            candidate_topics = [
                {'ItemId': 'A', 'Topic': 'Privacy-related content'},
                {'ItemId': 'B', 'Topic': 'Generic content'},
            ]

            # Step 1: Redact
            masked_data = redactor.redact_user_data(user_data)

            # Step 2: Ensemble evaluation
            all_results = []
            for evaluator in evaluators:
                try:
                    eval_results = evaluator.evaluate_interest(masked_data, candidate_topics)
                    all_results.append(eval_results)
                except Exception as e:
                    print(f"      ⚠️  Model {evaluator.model_name} error: {e}")

            # Step 3: Consensus
            if all_results:
                consensus = aggregator.aggregate_median(all_results)
                output_str = json.dumps(consensus)

                # Check for PII leakage
                leaked = False
                for entity_type in sample['pii_entities']:
                    # Check if any PII from source appears in output
                    if any(term.lower() in output_str.lower() for term in [entity_type, 'email', 'phone', 'name', 'address']):
                        leaked = True
                        break

                if leaked:
                    results['pii_leaked_count'] += 1
                    print(f"      ❌ PII LEAKED in output")
                else:
                    results['pii_protected_count'] += 1
                    print(f"      ✅ PII PROTECTED")

                # Count PII types
                for pii_type in sample['pii_entities']:
                    results['pii_types_found'][pii_type] += 1

            else:
                print(f"      ⚠️  All models failed")
                results['failed_samples'].append(sample['id'])

        except Exception as e:
            print(f"      ❌ Error: {e}")
            results['failed_samples'].append(sample['id'])

    # Calculate metrics
    results['pii_leak_rate'] = results['pii_leaked_count'] / results['samples_with_pii'] if results['samples_with_pii'] > 0 else 0
    results['protection_rate'] = results['pii_protected_count'] / results['samples_with_pii'] if results['samples_with_pii'] > 0 else 0

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate on public privacy datasets')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--output', default='public_dataset_results.json',
                       help='Output file (default: public_dataset_results.json)')

    args = parser.parse_args()

    # Check API key
    api_key = os.getenv('SAMBANOVA_API_KEY')
    if not api_key:
        print("❌ Error: SAMBANOVA_API_KEY not set!")
        print("\nSet your API key:")
        print("  export SAMBANOVA_API_KEY='your-key-here'")
        sys.exit(1)

    print("="*80)
    print("PUBLIC DATASET BENCHMARK - ai4privacy/pii-masking-200k")
    print("="*80)
    print(f"\n✓ API Key: {api_key[:20]}...")

    # Your 4-model ensemble
    model_names = [
        "gpt-oss-120b",
        "DeepSeek-V3.1",
        "Qwen3-32B",
        "DeepSeek-V3-0324",
    ]
    print(f"✓ Models: {', '.join(model_names)}")

    # Load dataset
    loader = DatasetLoader()
    samples = loader.load_ai4privacy(args.num_samples)

    # Run evaluation
    start_time = time.time()
    results = evaluate_on_dataset(samples, model_names, api_key)
    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n✓ Total samples evaluated: {results['total_samples']}")
    print(f"✓ Samples with PII: {results['samples_with_pii']}")
    print(f"✓ PII leaked: {results['pii_leaked_count']}")
    print(f"✓ PII protected: {results['pii_protected_count']}")
    print(f"✓ Failed samples: {len(results['failed_samples'])}")
    print(f"\n✓ PII Leak Rate: {results['pii_leak_rate']*100:.2f}%")
    print(f"✓ Protection Rate: {results['protection_rate']*100:.2f}%")
    print(f"\n✓ Time: {elapsed:.1f}s ({elapsed/results['total_samples']:.1f}s per sample)")

    print("\n📊 Top PII Types Found:")
    for pii_type, count in sorted(results['pii_types_found'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {pii_type}: {count}")

    # Save results
    results_with_config = {
        'config': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'ai4privacy/pii-masking-200k',
            'num_samples': args.num_samples,
            'models': model_names,
        },
        'results': {k: v for k, v in results.items() if k != 'pii_types_found'},
        'pii_types': dict(results['pii_types_found']),
    }

    with open(args.output, 'w') as f:
        json.dump(results_with_config, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
