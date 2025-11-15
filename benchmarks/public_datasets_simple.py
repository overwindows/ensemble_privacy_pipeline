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
from examples.llm_evaluators import TextMaskingEvaluator, check_pii_leakage


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
    Evaluate TEXT MASKING on real PII dataset samples.

    Task: Mask PII in text using ensemble LLMs
    Tests:
    1. PII Masking Rate: % of PII entities successfully masked
    2. PII Leakage: Do any PII entities appear in masked output?
    3. Utility: Is the masked text still coherent and useful?
    """
    print("\n" + "="*80)
    print("BENCHMARK: TEXT MASKING (PII Redaction)")
    print("="*80)
    print("\nTask: Mask PII in text samples from ai4privacy/pii-masking-200k")

    # Initialize evaluators for text masking
    evaluators = [TextMaskingEvaluator(model_name=model, api_key=api_key) for model in model_names]

    results = {
        'total_samples': len(samples),
        'total_pii_entities': 0,
        'pii_fully_masked_count': 0,
        'pii_leaked_count': 0,
        'pii_types_found': defaultdict(int),
        'failed_samples': [],
        'leaked_samples': [],
    }

    # Process each sample
    for i, sample in enumerate(samples, 1):
        print(f"\n  Sample {i}/{len(samples)}: {sample['id']}")
        print(f"    PII entities: {sample['pii_count']} types: {', '.join(sample['pii_entities'][:5])}")

        try:
            # Extract actual PII values from source text by comparing with masked version
            pii_values = []
            source_words = set(sample['source_text'].lower().split())
            masked_words = set(sample['masked_text'].lower().split())

            # Simple heuristic: words in source but not in masked are likely PII
            # (This is imperfect but sufficient for benchmark)
            potential_pii = source_words - masked_words
            pii_values = [word for word in potential_pii if len(word) > 2 and not word.startswith('[')]

            results['total_pii_entities'] += len(pii_values)

            # Step 1: Ask each model to mask the text
            masked_outputs = []
            for evaluator in evaluators:
                try:
                    # Mask the source text
                    masked_output = evaluator.mask_text(sample['source_text'][:1000])  # Limit to 1000 chars
                    masked_outputs.append(masked_output)
                except Exception as e:
                    print(f"      ⚠️  Model {evaluator.model_name} error: {e}")

            # Step 2: Ensemble consensus - use majority voting on masking
            if masked_outputs:
                # For simplicity, use the first model's output (could do consensus voting)
                final_masked = masked_outputs[0]

                # Step 3: Check for PII leakage in the masked output
                leakage_check = check_pii_leakage(final_masked, pii_values)

                if leakage_check['is_protected']:
                    results['pii_fully_masked_count'] += 1
                    print(f"      ✅ ALL PII MASKED ({len(pii_values)} entities protected)")
                else:
                    results['pii_leaked_count'] += 1
                    results['leaked_samples'].append({
                        'id': sample['id'],
                        'leaked_entities': leakage_check['leaked_entities'][:3],
                        'leak_rate': leakage_check['leakage_rate']
                    })
                    print(f"      ❌ PII LEAKED: {leakage_check['leaked_count']}/{len(pii_values)} entities")
                    print(f"         Examples: {leakage_check['leaked_entities'][:3]}")

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
    results['masking_success_rate'] = results['pii_fully_masked_count'] / results['total_samples'] if results['total_samples'] > 0 else 0
    results['leakage_rate'] = results['pii_leaked_count'] / results['total_samples'] if results['total_samples'] > 0 else 0

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
    api_key = os.getenv('LLM_API_KEY')
    if not api_key:
        print("❌ Error: LLM_API_KEY not set!")
        print("\nSet your API key:")
        print("  export LLM_API_KEY='your-key-here'")
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
    print(f"✓ Total PII entities: {results['total_pii_entities']}")
    print(f"✓ Samples with full masking: {results['pii_fully_masked_count']}")
    print(f"✓ Samples with PII leaked: {results['pii_leaked_count']}")
    print(f"✓ Failed samples: {len(results['failed_samples'])}")
    print(f"\n✓ Masking Success Rate: {results['masking_success_rate']*100:.2f}%")
    print(f"✓ PII Leakage Rate: {results['leakage_rate']*100:.2f}%")
    print(f"\n✓ Time: {elapsed:.1f}s ({elapsed/results['total_samples']:.1f}s per sample)")

    print("\n📊 Top PII Types Found:")
    for pii_type, count in sorted(results['pii_types_found'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {pii_type}: {count}")

    # Save results
    results_with_config = {
        'config': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'task': 'Text Masking (PII Redaction)',
            'dataset': 'ai4privacy/pii-masking-200k',
            'num_samples': args.num_samples,
            'models': model_names,
        },
        'results': {
            'total_samples': results['total_samples'],
            'total_pii_entities': results['total_pii_entities'],
            'masking_success_rate': results['masking_success_rate'],
            'leakage_rate': results['leakage_rate'],
            'pii_fully_masked_count': results['pii_fully_masked_count'],
            'pii_leaked_count': results['pii_leaked_count'],
            'failed_count': len(results['failed_samples']),
        },
        'pii_types': dict(results['pii_types_found']),
        'leaked_samples': results['leaked_samples'][:10],  # First 10 for review
    }

    with open(args.output, 'w') as f:
        json.dump(results_with_config, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
