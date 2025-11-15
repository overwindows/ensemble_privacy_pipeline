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
    # With local PUPA dataset
    python3 benchmarks/pupa_benchmark.py --dataset-path /path/to/pupa.json --num-samples 100

    # With WildChat simulation (if PUPA not available)
    python3 benchmarks/pupa_benchmark.py --simulate --num-samples 50
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.privacy_core import PrivacyRedactor, ConsensusAggregator
from examples.real_llm_example import RealLLMEvaluator


# ============================================================================
# PUPA DATASET LOADER
# ============================================================================

class PUPADatasetLoader:
    """Load and prepare PUPA dataset or simulate WildChat-style data."""

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
            print(f"\nAlternatively, use --simulate flag to generate synthetic samples")
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

    def simulate_wildchat_samples(self, num_samples: int = 50) -> List[Dict]:
        """
        Simulate WildChat-style samples with realistic PII.

        Creates samples in the 3 PUPA categories:
        1. Job/Visa Applications
        2. Financial/Corporate Info
        3. Quoted Emails/Messages
        """
        print(f"\n🔨 Simulating {num_samples} WildChat-style samples...")
        print(f"   (Based on PUPA categories)")

        samples = []

        categories = [
            'Job, Visa, & Other Applications',
            'Financial and Corporate Info',
            'Quoted Emails and Messages'
        ]

        for i in range(num_samples):
            category = categories[i % 3]

            if category == 'Job, Visa, & Other Applications':
                sample = self._generate_job_sample(i)
            elif category == 'Financial and Corporate Info':
                sample = self._generate_financial_sample(i)
            else:
                sample = self._generate_email_sample(i)

            sample['id'] = f'simulated_{i}'
            sample['pii_category'] = category
            samples.append(sample)

        print(f"✓ Generated {len(samples)} simulated samples")

        # Show distribution
        category_counts = defaultdict(int)
        for s in samples:
            category_counts[s['pii_category']] += 1

        print(f"\n   Category distribution:")
        for cat, count in category_counts.items():
            print(f"     - {cat}: {count} ({count/len(samples)*100:.1f}%)")

        return samples

    def _generate_job_sample(self, idx: int) -> Dict:
        """Generate job/visa application sample."""
        names = ["John Smith", "Sarah Johnson", "Michael Chen", "Emily Davis", "Robert Wilson"]
        companies = ["TechCorp Inc", "Global Solutions Ltd", "Innovation Labs", "DataSys Corp"]
        positions = ["Software Engineer", "Data Scientist", "Product Manager", "Senior Developer"]

        name = names[idx % len(names)]
        company = companies[idx % len(companies)]
        position = positions[idx % len(positions)]
        email = f"{name.lower().replace(' ', '.')}@email.com"

        user_prompt = f"""I'm applying for a {position} position at {company}.
Can you help me write a cover letter? Here's my information:
Name: {name}
Email: {email}
Current role: Senior Engineer at Microsoft
Years of experience: 5 years

I want to emphasize my experience with Python, machine learning, and cloud infrastructure."""

        return {
            'user_prompt': user_prompt,
            'assistant_response': '[Simulated response about cover letter writing]',
            'pii_units': [name, company, position, email, 'Microsoft', '5 years'],
        }

    def _generate_financial_sample(self, idx: int) -> Dict:
        """Generate financial/corporate info sample."""
        companies = ["Acme Corp", "TechStart Inc", "DataFlow Systems", "CloudNine Ltd"]
        amounts = ["$250,000", "$1.5M", "$750K", "$500,000"]
        investors = ["Sequoia Capital", "Andreessen Horowitz", "Y Combinator", "Benchmark"]

        company = companies[idx % len(companies)]
        amount = amounts[idx % len(amounts)]
        investor = investors[idx % len(investors)]

        user_prompt = f"""I need help drafting a term sheet for my startup {company}.
We're raising {amount} in seed funding from {investor}.

Key terms:
- Valuation: $5M pre-money
- Board seats: 2 for founders, 1 for investor
- Liquidation preference: 1x non-participating

Can you help structure this properly?"""

        return {
            'user_prompt': user_prompt,
            'assistant_response': '[Simulated response about term sheet]',
            'pii_units': [company, amount, investor, '$5M', 'seed funding'],
        }

    def _generate_email_sample(self, idx: int) -> Dict:
        """Generate quoted email/message sample."""
        names = ["Alice Brown", "Bob Taylor", "Carol White", "David Green"]
        emails = ["alice.b@company.com", "bob.t@corp.com", "carol.w@startup.io", "david.g@tech.com"]

        name = names[idx % len(names)]
        email = emails[idx % len(emails)]

        user_prompt = f"""I received this email from my colleague {name} ({email}):

"Hi team,

I wanted to update you on the Q4 roadmap. We need to prioritize:
1. API migration to v2.0
2. Database scaling for new customers
3. Security audit for SOC2 compliance

Let me know if you have concerns. Meeting scheduled for next Tuesday at 2pm.

Best,
{name}
Engineering Manager"

Can you help me draft a response about prioritization concerns?"""

        return {
            'user_prompt': user_prompt,
            'assistant_response': '[Simulated response about email reply]',
            'pii_units': [name, email, 'Q4', 'Tuesday at 2pm', 'Engineering Manager'],
        }


# ============================================================================
# PUPA EVALUATION
# ============================================================================

def evaluate_pupa_benchmark(samples: List[Dict], model_names: List[str], api_key: str) -> Dict:
    """
    Evaluate privacy pipeline on PUPA benchmark.

    Metrics:
    1. Privacy Leakage: % of PII units exposed in output
    2. Quality Preservation: Ability to provide relevant responses
    3. Category Breakdown: Performance per PII category
    """
    print("\n" + "="*80)
    print("PUPA BENCHMARK EVALUATION")
    print("Private User Prompt Annotations Dataset (NAACL 2025)")
    print("="*80)

    print(f"\n✓ Samples: {len(samples)}")
    print(f"✓ Models: {len(model_names)}")

    # Initialize pipeline
    redactor = PrivacyRedactor()
    aggregator = ConsensusAggregator()
    evaluators = [RealLLMEvaluator(model_name=model, api_key=api_key) for model in model_names]

    results = {
        'total_samples': len(samples),
        'pii_units_total': 0,
        'pii_units_leaked': 0,
        'pii_leakage_rate': 0.0,
        'quality_preserved_count': 0,
        'quality_preservation_rate': 0.0,
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

        # Convert to pipeline format
        user_data = {
            'raw_queries': [sample['user_prompt']],
            'demographics': {}
        }

        # Dummy candidate topic (for evaluation task)
        candidate_topics = [
            {'ItemId': 'A', 'Topic': 'User assistance and information request'},
            {'ItemId': 'B', 'Topic': 'Unrelated topic'},
        ]

        try:
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
                output_str = json.dumps(consensus) + json.dumps(masked_data)

                # Check for PII leakage
                leaked_pii = []
                for pii_unit in sample['pii_units']:
                    if pii_unit and pii_unit.lower() in output_str.lower():
                        leaked_pii.append(pii_unit)

                results['pii_units_total'] += len(sample['pii_units'])
                results['pii_units_leaked'] += len(leaked_pii)

                # Update category stats
                category = sample['pii_category']
                results['category_breakdown'][category]['samples'] += 1
                results['category_breakdown'][category]['pii_total'] += len(sample['pii_units'])
                results['category_breakdown'][category]['pii_leaked'] += len(leaked_pii)

                if leaked_pii:
                    results['leaked_samples'].append({
                        'id': sample['id'],
                        'category': category,
                        'leaked_pii': leaked_pii,
                        'leak_rate': len(leaked_pii) / len(sample['pii_units'])
                    })
                    print(f"      ❌ PII LEAKED: {len(leaked_pii)}/{len(sample['pii_units'])} units")
                    print(f"         {leaked_pii[:3]}...")
                else:
                    results['protected_samples'].append(sample['id'])
                    print(f"      ✅ ALL PII PROTECTED ({len(sample['pii_units'])} units)")

                # Quality check (simple heuristic: score > 0.3)
                if consensus and consensus[0]['QualityScore'] > 0.3:
                    results['quality_preserved_count'] += 1

            else:
                print(f"      ⚠️  All models failed")

        except Exception as e:
            print(f"      ❌ Error: {e}")

    # Calculate final metrics
    if results['pii_units_total'] > 0:
        results['pii_leakage_rate'] = results['pii_units_leaked'] / results['pii_units_total']

    results['quality_preservation_rate'] = results['quality_preserved_count'] / results['total_samples']

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
            'quality_preservation': results['quality_preservation_rate'],
            'privacy_leakage': results['pii_leakage_rate'],
        },
        'papillon_baseline': papillon_baseline,
        'improvements': {
            'quality_diff': results['quality_preservation_rate'] - papillon_baseline['quality_preservation'],
            'leakage_diff': papillon_baseline['privacy_leakage'] - results['pii_leakage_rate'],
        }
    }

    print("\n📊 Comparison Table:")
    print("-"*80)
    print(f"{'Metric':<30} {'Your Ensemble':<20} {'PAPILLON':<20}")
    print("-"*80)
    print(f"{'Quality Preservation':<30} {comparison['your_approach']['quality_preservation']*100:>8.1f}% {' '*10} {papillon_baseline['quality_preservation']*100:>8.1f}%")
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
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to PUPA dataset JSON file')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulate WildChat-style samples (if PUPA not available)')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--output', default='pupa_benchmark_results.json',
                       help='Output file (default: pupa_benchmark_results.json)')

    args = parser.parse_args()

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

    if args.dataset_path:
        samples = loader.load_pupa_dataset(args.dataset_path, args.num_samples)
    elif args.simulate:
        samples = loader.simulate_wildchat_samples(args.num_samples)
    else:
        print("\n❌ Error: Must specify either --dataset-path or --simulate")
        print("\nUsage:")
        print("  # With real PUPA dataset:")
        print("  python3 benchmarks/pupa_benchmark.py --dataset-path /path/to/pupa.json")
        print("\n  # With simulated data:")
        print("  python3 benchmarks/pupa_benchmark.py --simulate --num-samples 50")
        sys.exit(1)

    # Run evaluation
    start_time = time.time()
    results = evaluate_pupa_benchmark(samples, model_names, api_key)
    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "="*80)
    print("PUPA BENCHMARK RESULTS")
    print("="*80)

    print(f"\n✓ Total samples: {results['total_samples']}")
    print(f"✓ Total PII units: {results['pii_units_total']}")
    print(f"✓ PII units leaked: {results['pii_units_leaked']}")
    print(f"✓ PII units protected: {results['pii_units_total'] - results['pii_units_leaked']}")

    print(f"\n📊 Key Metrics:")
    print(f"   Privacy Leakage Rate: {results['pii_leakage_rate']*100:.2f}%")
    print(f"   Quality Preservation: {results['quality_preservation_rate']*100:.1f}%")

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
            'dataset': 'PUPA (NAACL 2025)' if args.dataset_path else 'WildChat-simulated',
            'num_samples': args.num_samples,
            'models': model_names,
        },
        'results': {
            'pii_leakage_rate': results['pii_leakage_rate'],
            'quality_preservation_rate': results['quality_preservation_rate'],
            'pii_units_total': results['pii_units_total'],
            'pii_units_leaked': results['pii_units_leaked'],
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
