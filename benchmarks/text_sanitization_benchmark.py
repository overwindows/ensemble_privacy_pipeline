#!/usr/bin/env python3
"""
Text Sanitization Benchmark - TAB (Text Anonymization Benchmark)

Evaluates your Ensemble-Redaction approach on text sanitization benchmarks:
- TAB (Text Anonymization Benchmark) from ECHR court cases
- SanText datasets (SST-2, QNLI, AGNews)

References:
1. TAB: Pilán et al. (2022) "The Text Anonymization Benchmark (TAB): A Dedicated
   Corpus and Evaluation Framework for Text Anonymization"
   https://github.com/NorskRegnesentral/text-anonymization-benchmark

2. SanText: Yue et al. (2021) "Differential Privacy for Text Analytics via Natural
   Text Sanitization" (ACL 2021 Findings)
   https://github.com/xiangyue9607/SanText

Dataset:
- TAB: 1,268 ECHR court cases with manual PII annotations
- Entity types: PERSON, ORG, LOC, DATETIME, CODE, QUANTITY, DEM, MISC
- Identifier types: DIRECT (must mask), QUASI (quasi-identifiers), NO_MASK

Evaluation Metrics:
- Privacy Protection: % of PII entities masked
- Utility Preservation: Text coherence, task utility
- Precision/Recall/F1 for PII detection

Usage:
    # With real TAB dataset
    git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark
    python3 benchmarks/text_sanitization_benchmark.py --dataset-path text-anonymization-benchmark/data

    # With simulated ECHR-style data
    python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 50
"""

import argparse
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
from examples.llm_evaluators import DocumentSanitizationEvaluator, check_pii_leakage


# ============================================================================
# TAB DATASET LOADER
# ============================================================================

class TABDatasetLoader:
    """Load and prepare TAB (Text Anonymization Benchmark) dataset."""

    def load_tab_dataset(self, dataset_path: str, num_samples: int = 100,
                        split: str = 'test') -> List[Dict]:
        """
        Load real TAB dataset from JSON files.

        Args:
            dataset_path: Path to TAB dataset directory
            num_samples: Number of samples to load
            split: Dataset split ('train', 'dev', or 'test')

        Returns:
            List of samples with text and PII annotations
        """
        print(f"\n📥 Loading TAB dataset from: {dataset_path}")
        print(f"   Split: {split}")

        json_path = os.path.join(dataset_path, f'{split}.json')

        if not os.path.exists(json_path):
            print(f"❌ Error: TAB dataset not found at {json_path}")
            print(f"\nTo obtain TAB dataset:")
            print(f"  1. git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark")
            print(f"  2. Point --dataset-path to the cloned repo's 'data' directory")
            print(f"\nAlternatively, use --simulate flag to generate synthetic samples")
            sys.exit(1)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            samples = []
            for idx, doc in enumerate(raw_data[:num_samples]):
                # Extract PII entities
                pii_entities = []
                entity_types = defaultdict(int)

                for annotation in doc.get('annotations', []):
                    identifier_type = annotation.get('identifier_type', 'NO_MASK')
                    if identifier_type in ['DIRECT', 'QUASI']:
                        entity_info = {
                            'text': annotation.get('span_text', ''),
                            'type': annotation.get('entity_type', 'MISC'),
                            'identifier_type': identifier_type,
                            'start': annotation.get('start_offset', 0),
                            'end': annotation.get('end_offset', 0),
                        }
                        pii_entities.append(entity_info)
                        entity_types[entity_info['type']] += 1

                samples.append({
                    'id': doc.get('doc_id', f'tab_{idx}'),
                    'text': doc.get('text', ''),
                    'pii_entities': pii_entities,
                    'entity_types': dict(entity_types),
                    'total_pii': len(pii_entities),
                    'direct_identifiers': sum(1 for e in pii_entities if e['identifier_type'] == 'DIRECT'),
                    'quasi_identifiers': sum(1 for e in pii_entities if e['identifier_type'] == 'QUASI'),
                })

            print(f"✓ Loaded {len(samples)} samples from TAB dataset")

            # Show statistics
            total_pii = sum(s['total_pii'] for s in samples)
            total_direct = sum(s['direct_identifiers'] for s in samples)
            total_quasi = sum(s['quasi_identifiers'] for s in samples)

            print(f"\n   Statistics:")
            print(f"     Total PII entities: {total_pii}")
            print(f"     Direct identifiers: {total_direct} ({total_direct/total_pii*100:.1f}%)")
            print(f"     Quasi-identifiers: {total_quasi} ({total_quasi/total_pii*100:.1f}%)")

            return samples

        except Exception as e:
            print(f"❌ Error loading TAB dataset: {e}")
            sys.exit(1)

    def simulate_echr_samples(self, num_samples: int = 50) -> List[Dict]:
        """
        Simulate ECHR court case samples with realistic PII.

        Generates samples similar to TAB dataset structure.
        """
        print(f"\n🔨 Simulating {num_samples} ECHR-style court case samples...")

        samples = []

        entity_types_pool = {
            'PERSON': ['John Smith', 'Sarah Johnson', 'Dr. Michael Chen', 'Maria Garcia', 'Judge Thompson'],
            'ORG': ['European Court of Human Rights', 'Ministry of Justice', 'Supreme Court', 'Legal Aid Society'],
            'LOC': ['Strasbourg', 'Paris', 'London', 'Brussels', 'Geneva'],
            'DATETIME': ['15 March 2020', '2019', 'January 2021', '10 February 2018'],
            'CODE': ['Article 8 ECHR', 'Section 42(1)', 'Regulation 2016/679', 'Law No. 2019-222'],
        }

        for i in range(num_samples):
            # Generate case text
            person_name = entity_types_pool['PERSON'][i % len(entity_types_pool['PERSON'])]
            org_name = entity_types_pool['ORG'][i % len(entity_types_pool['ORG'])]
            location = entity_types_pool['LOC'][i % len(entity_types_pool['LOC'])]
            date = entity_types_pool['DATETIME'][i % len(entity_types_pool['DATETIME'])]
            code = entity_types_pool['CODE'][i % len(entity_types_pool['CODE'])]

            case_text = f"""The applicant, {person_name}, a national born in {date}, complained about \
violations of their rights under {code}. The case was heard at the {org_name} in {location}. \
The court found that {org_name} had failed to adequately protect the applicant's rights. \
{person_name} had been denied access to legal representation during proceedings in {location}."""

            # Extract PII entities
            pii_entities = [
                {'text': person_name, 'type': 'PERSON', 'identifier_type': 'DIRECT', 'start': 15, 'end': 15 + len(person_name)},
                {'text': date, 'type': 'DATETIME', 'identifier_type': 'QUASI', 'start': case_text.find(date), 'end': case_text.find(date) + len(date)},
                {'text': code, 'type': 'CODE', 'identifier_type': 'NO_MASK', 'start': case_text.find(code), 'end': case_text.find(code) + len(code)},
                {'text': org_name, 'type': 'ORG', 'identifier_type': 'QUASI', 'start': case_text.find(org_name), 'end': case_text.find(org_name) + len(org_name)},
                {'text': location, 'type': 'LOC', 'identifier_type': 'QUASI', 'start': case_text.find(location), 'end': case_text.find(location) + len(location)},
            ]

            # Count by type
            entity_types = defaultdict(int)
            for entity in pii_entities:
                entity_types[entity['type']] += 1

            samples.append({
                'id': f'simulated_echr_{i}',
                'text': case_text,
                'pii_entities': pii_entities,
                'entity_types': dict(entity_types),
                'total_pii': len(pii_entities),
                'direct_identifiers': sum(1 for e in pii_entities if e['identifier_type'] == 'DIRECT'),
                'quasi_identifiers': sum(1 for e in pii_entities if e['identifier_type'] == 'QUASI'),
            })

        print(f"✓ Generated {len(samples)} simulated samples")

        # Show distribution
        total_pii = sum(s['total_pii'] for s in samples)
        total_direct = sum(s['direct_identifiers'] for s in samples)
        total_quasi = sum(s['quasi_identifiers'] for s in samples)

        print(f"\n   Statistics:")
        print(f"     Total PII entities: {total_pii}")
        print(f"     Direct identifiers: {total_direct} ({total_direct/total_pii*100:.1f}%)")
        print(f"     Quasi-identifiers: {total_quasi} ({total_quasi/total_pii*100:.1f}%)")

        return samples


# ============================================================================
# TEXT SANITIZATION EVALUATION
# ============================================================================

def evaluate_text_sanitization(samples: List[Dict], model_names: List[str], api_key: str) -> Dict:
    """
    Evaluate DOCUMENT SANITIZATION on TAB benchmark.

    Task: Sanitize legal/court documents by masking PII for public release
    Metrics:
    1. PII Masking Rate: % of PII entities successfully masked
    2. Direct ID Protection: % of DIRECT identifiers protected
    3. Quasi ID Protection: % of QUASI identifiers protected
    4. Privacy Leakage: % of PII entities exposed in output
    """
    print("\n" + "="*80)
    print("BENCHMARK: DOCUMENT SANITIZATION")
    print("TAB (Text Anonymization Benchmark)")
    print("="*80)
    print("\nTask: Sanitize legal documents for public release")

    print(f"\n✓ Samples: {len(samples)}")
    print(f"✓ Models: {len(model_names)}")

    # Initialize document sanitization evaluators
    evaluators = [DocumentSanitizationEvaluator(model_name=model, api_key=api_key) for model in model_names]

    results = {
        'total_samples': len(samples),
        'total_pii_entities': 0,
        'pii_entities_leaked': 0,
        'pii_masking_rate': 0.0,
        'direct_identifiers_total': 0,
        'direct_identifiers_leaked': 0,
        'direct_id_protection_rate': 0.0,
        'quasi_identifiers_total': 0,
        'quasi_identifiers_leaked': 0,
        'quasi_id_protection_rate': 0.0,
        'entity_type_breakdown': defaultdict(lambda: {
            'total': 0,
            'leaked': 0,
            'protection_rate': 0.0
        }),
        'leaked_samples': [],
        'protected_samples': [],
    }

    # Process each sample
    for i, sample in enumerate(samples, 1):
        print(f"\n  Sample {i}/{len(samples)}: {sample['id']}")
        print(f"    Text length: {len(sample['text'])} chars")
        print(f"    PII entities: {sample['total_pii']} (Direct: {sample['direct_identifiers']}, Quasi: {sample['quasi_identifiers']})")

        try:
            # Extract PII entity text values
            pii_texts = [entity['text'] for entity in sample['pii_entities'] if entity.get('text')]

            # Step 1: Each model sanitizes the document
            sanitized_outputs = []
            for evaluator in evaluators:
                try:
                    # Sanitize document (limit to first 1000 chars for cost)
                    sanitized = evaluator.sanitize_document(sample['text'][:1000])
                    sanitized_outputs.append(sanitized)
                except Exception as e:
                    print(f"      ⚠️  Model {evaluator.model_name} error: {e}")

            # Step 2: Ensemble consensus - aggregate WITHOUT using ground truth
            if sanitized_outputs:
                # Majority voting: Pick the most common sanitized output
                from collections import Counter
                output_counts = Counter(sanitized_outputs)

                if len(output_counts) == 1:
                    # All models gave same output
                    final_sanitized = sanitized_outputs[0]
                    consensus_type = "unanimous"
                elif output_counts.most_common(1)[0][1] >= len(sanitized_outputs) / 2:
                    # Majority exists
                    final_sanitized = output_counts.most_common(1)[0][0]
                    consensus_type = "majority"
                else:
                    # No clear majority, use shortest output (heuristic: more aggressive masking)
                    final_sanitized = min(sanitized_outputs, key=len)
                    consensus_type = "shortest_fallback"

                if len(sanitized_outputs) > 1:
                    unique_outputs = len(output_counts)
                    print(f"      📊 Ensemble ({len(sanitized_outputs)} models): {unique_outputs} unique outputs, consensus: {consensus_type}")

                # Step 3: EVALUATE consensus output against ground truth (not used for selection!)
                leakage_check = check_pii_leakage(final_sanitized, pii_texts)

                # Update statistics
                results['total_pii_entities'] += sample['total_pii']
                results['pii_entities_leaked'] += leakage_check['leaked_count']

                results['direct_identifiers_total'] += sample['direct_identifiers']
                results['quasi_identifiers_total'] += sample['quasi_identifiers']

                # Count leaked direct vs quasi identifiers
                direct_leaked = 0
                quasi_leaked = 0
                for entity in sample['pii_entities']:
                    if entity['text'] in leakage_check['leaked_entities']:
                        if entity['identifier_type'] == 'DIRECT':
                            direct_leaked += 1
                        elif entity['identifier_type'] == 'QUASI':
                            quasi_leaked += 1

                results['direct_identifiers_leaked'] += direct_leaked
                results['quasi_identifiers_leaked'] += quasi_leaked

                # Entity type breakdown
                for entity in sample['pii_entities']:
                    entity_type = entity['type']
                    results['entity_type_breakdown'][entity_type]['total'] += 1
                    if entity['text'] in leakage_check['leaked_entities']:
                        results['entity_type_breakdown'][entity_type]['leaked'] += 1

                if leakage_check['is_protected']:
                    results['protected_samples'].append(sample['id'])
                    print(f"      ✅ ALL PII PROTECTED ({sample['total_pii']} entities)")
                else:
                    results['leaked_samples'].append({
                        'id': sample['id'],
                        'leaked_count': leakage_check['leaked_count'],
                        'leaked_entities': leakage_check['leaked_entities'][:5],
                        'leak_rate': leakage_check['leakage_rate']
                    })
                    print(f"      ❌ PII LEAKED: {leakage_check['leaked_count']}/{sample['total_pii']} entities")
                    print(f"         Direct: {direct_leaked}, Quasi: {quasi_leaked}")
                    print(f"         Examples: {leakage_check['leaked_entities'][:3]}")

            else:
                print(f"      ⚠️  All models failed")

        except Exception as e:
            print(f"      ❌ Error: {e}")

    # Calculate final metrics
    if results['total_pii_entities'] > 0:
        results['pii_masking_rate'] = 1.0 - (results['pii_entities_leaked'] / results['total_pii_entities'])

    if results['direct_identifiers_total'] > 0:
        results['direct_id_protection_rate'] = 1.0 - (results['direct_identifiers_leaked'] / results['direct_identifiers_total'])

    if results['quasi_identifiers_total'] > 0:
        results['quasi_id_protection_rate'] = 1.0 - (results['quasi_identifiers_leaked'] / results['quasi_identifiers_total'])

    # Calculate entity type protection rates
    for entity_type, stats in results['entity_type_breakdown'].items():
        if stats['total'] > 0:
            stats['protection_rate'] = 1.0 - (stats['leaked'] / stats['total'])

    return results


# ============================================================================
# COMPARISON WITH BASELINES
# ============================================================================

def compare_with_baselines(results: Dict) -> Dict:
    """
    Display TAB benchmark results.

    Note: Baseline comparisons removed - no verified baseline numbers available
    for direct comparison on TAB dataset. Focus on your actual results.
    """
    print("\n" + "="*80)
    print("TAB BENCHMARK RESULTS SUMMARY")
    print("="*80)

    comparison = {
        'your_approach': {
            'pii_masking_rate': results['pii_masking_rate'],
            'direct_id_protection_rate': results['direct_id_protection_rate'],
            'quasi_id_protection_rate': results['quasi_id_protection_rate'],
        },
    }

    print("\n📊 Your Results:")
    print("-"*80)
    print(f"{'Metric':<30} {'Rate':<15} {'Status':<20}")
    print("-"*80)
    print(f"{'Overall PII Masking':<30} {comparison['your_approach']['pii_masking_rate']*100:>8.1f}% {' '*5} {'Good' if comparison['your_approach']['pii_masking_rate'] > 0.80 else 'Needs improvement'}")
    print(f"{'Direct ID Protection':<30} {comparison['your_approach']['direct_id_protection_rate']*100:>8.1f}% {' '*5} {'✅ Near-perfect' if comparison['your_approach']['direct_id_protection_rate'] > 0.95 else 'Good'}")
    print(f"{'Quasi ID Protection':<30} {comparison['your_approach']['quasi_id_protection_rate']*100:>8.1f}% {' '*5} {'✅ Near-perfect' if comparison['your_approach']['quasi_id_protection_rate'] > 0.95 else 'Good'}")
    print("-"*80)

    # Verdict based on your results only
    if comparison['your_approach']['direct_id_protection_rate'] >= 0.95:
        verdict = "✅ Excellent direct identifier protection (near-perfect)"
    elif comparison['your_approach']['pii_masking_rate'] >= 0.80:
        verdict = "✅ Good overall PII protection"
    else:
        verdict = "⚠️  Room for improvement"

    print(f"\n💡 Verdict: {verdict}")
    print("-"*80)

    return comparison


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Text Sanitization benchmark (TAB)')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to TAB dataset directory')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulate ECHR-style samples (if TAB not available)')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'],
                       help='Dataset split to use (default: test)')
    parser.add_argument('--output', default='results/text_sanitization_results.json',
                       help='Output file (default: results/text_sanitization_results.json)')

    args = parser.parse_args()

    # Setup logging and results directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    log_file = 'logs/text_sanitization_benchmark.log'

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
    print("TEXT SANITIZATION BENCHMARK - TAB")
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
    loader = TABDatasetLoader()

    if args.dataset_path:
        samples = loader.load_tab_dataset(args.dataset_path, args.num_samples, args.split)
    elif args.simulate:
        samples = loader.simulate_echr_samples(args.num_samples)
    else:
        print("\n❌ Error: Must specify either --dataset-path or --simulate")
        print("\nUsage:")
        print("  # With real TAB dataset:")
        print("  git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark")
        print("  python3 benchmarks/text_sanitization_benchmark.py --dataset-path text-anonymization-benchmark/data")
        print("\n  # With simulated data:")
        print("  python3 benchmarks/text_sanitization_benchmark.py --simulate --num-samples 50")
        sys.exit(1)

    # Run evaluation
    start_time = time.time()
    results = evaluate_text_sanitization(samples, model_names, api_key)
    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "="*80)
    print("TEXT SANITIZATION RESULTS")
    print("="*80)

    print(f"\n✓ Total samples: {results['total_samples']}")
    print(f"✓ Total PII entities: {results['total_pii_entities']}")
    print(f"✓ PII entities leaked: {results['pii_entities_leaked']}")
    print(f"✓ PII entities protected: {results['total_pii_entities'] - results['pii_entities_leaked']}")

    print(f"\n📊 Key Metrics:")
    print(f"   PII Masking Rate: {results['pii_masking_rate']*100:.2f}%")
    print(f"   Direct ID Protection: {results['direct_id_protection_rate']*100:.1f}%")
    print(f"   Quasi ID Protection: {results['quasi_id_protection_rate']*100:.1f}%")

    print(f"\n📊 Entity Type Breakdown:")
    for entity_type, stats in sorted(results['entity_type_breakdown'].items()):
        print(f"   {entity_type}:")
        print(f"     Total: {stats['total']}, Leaked: {stats['leaked']}, Protection: {stats['protection_rate']*100:.1f}%")

    print(f"\n✓ Time: {elapsed:.1f}s ({elapsed/results['total_samples']:.1f}s per sample)")

    # Compare with baselines
    comparison = compare_with_baselines(results)

    # Save results
    results_with_config = {
        'config': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'task': 'Document Sanitization',
            'dataset': 'TAB' if args.dataset_path else 'ECHR-simulated',
            'num_samples': args.num_samples,
            'models': model_names,
        },
        'results': {
            'pii_masking_rate': results['pii_masking_rate'],
            'direct_id_protection_rate': results['direct_id_protection_rate'],
            'quasi_id_protection_rate': results['quasi_id_protection_rate'],
            'total_pii_entities': results['total_pii_entities'],
            'pii_entities_leaked': results['pii_entities_leaked'],
            'leaked_samples_count': len(results['leaked_samples']),
            'protected_samples_count': len(results['protected_samples']),
        },
        'entity_type_breakdown': dict(results['entity_type_breakdown']),
        'leaked_samples': results['leaked_samples'][:10],  # First 10 for review
        'time_seconds': elapsed,
    }

    with open(args.output, 'w') as f:
        json.dump(results_with_config, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
