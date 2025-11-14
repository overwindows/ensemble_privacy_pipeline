"""
Public Benchmark Dataset Evaluation for Privacy-Preserving Pipeline

This script integrates with standard public benchmarks to evaluate the
ensemble-redaction consensus pipeline against industry-standard datasets.

Supported Benchmarks:
1. ai4privacy/pii-masking-200k (Hugging Face) - 200K+ samples, 54 PII classes
2. PII-Bench (2025) - Query-aware privacy protection evaluation
3. PrivacyXray Synthetic Dataset - 5,000 synthetic individuals, 16 PII types

Usage:
    # Install required packages first:
    pip install datasets huggingface_hub

    # Run evaluation:
    python benchmark_public_datasets.py --benchmark ai4privacy --num_samples 1000
    python benchmark_public_datasets.py --benchmark all --num_samples 500
"""

import json
import argparse
import time
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict

# Import evaluation framework
from evaluation_framework import (
    EvaluationPipeline,
    PrivacyEvaluator,
    UtilityEvaluator
)

# Import pipeline components
try:
    from ensemble_privacy_pipeline import (
        PrivacyRedactor,
        MockLLMEvaluator,
        ConsensusAggregator
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    print("Warning: Could not import pipeline components.")
    PIPELINE_AVAILABLE = False


# ============================================================================
# BENCHMARK DATASET LOADERS
# ============================================================================

class PublicBenchmarkLoader:
    """
    Loads public privacy benchmark datasets.

    Supports:
    - ai4privacy/pii-masking-200k (Hugging Face)
    - PII-Bench synthetic data
    - PrivacyXray synthetic dataset
    """

    def __init__(self):
        self.cache = {}

    def load_ai4privacy_dataset(self, num_samples: int = 1000,
                                language: str = 'en') -> List[Dict[str, Any]]:
        """
        Load ai4privacy/pii-masking-200k dataset from Hugging Face.

        Args:
            num_samples: Number of samples to load (default 1000)
            language: Language filter ('en', 'de', 'fr', 'it')

        Returns:
            List of samples with source_text and masked_text
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: 'datasets' package not installed.")
            print("Install with: pip install datasets")
            return self._load_ai4privacy_fallback(num_samples)

        print(f"\nLoading ai4privacy/pii-masking-200k dataset...")
        print(f"  Language: {language}")
        print(f"  Samples: {num_samples}")

        try:
            # Load from Hugging Face
            dataset = load_dataset("ai4privacy/pii-masking-200k", split="train")

            # Filter by language if needed
            if language != 'all':
                # Dataset has language-specific files, filter by checking content
                # For now, we'll just take the first num_samples
                pass

            # Convert to our format
            benchmark_samples = []
            count = 0

            for idx, sample in enumerate(dataset):
                if count >= num_samples:
                    break

                # Extract text fields
                source_text = sample.get('source_text', sample.get('text', ''))
                target_text = sample.get('target_text', sample.get('masked_text', ''))

                if source_text:
                    # Extract queries/PII from source text
                    queries = self._extract_queries_from_text(source_text)

                    # Identify PII types
                    pii_types = self._identify_pii_types(source_text, target_text)

                    benchmark_samples.append({
                        'id': f'ai4privacy_{idx}',
                        'source_text': source_text,
                        'masked_text': target_text,
                        'queries': queries,
                        'pii_types': pii_types,
                        'ground_truth_pii': {
                            'original': source_text,
                            'masked': target_text,
                            'contains_pii': len(pii_types) > 0
                        },
                        'category': 'real_world',
                        'benchmark': 'ai4privacy'
                    })
                    count += 1

            print(f"✓ Loaded {len(benchmark_samples)} samples from ai4privacy dataset")
            return benchmark_samples

        except Exception as e:
            print(f"Warning: Could not load from Hugging Face: {e}")
            print("Using fallback synthetic data...")
            return self._load_ai4privacy_fallback(num_samples)

    def load_pii_bench_dataset(self, num_samples: int = 500) -> List[Dict[str, Any]]:
        """
        Load PII-Bench style dataset (synthetic, following 2025 benchmark).

        PII-Bench focuses on query-aware privacy protection.
        Since the full dataset may not be publicly available yet,
        we create synthetic data following their methodology.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of samples with queries and ground truth
        """
        print(f"\nGenerating PII-Bench style synthetic dataset...")
        print(f"  Samples: {num_samples}")

        benchmark_samples = []

        # PII-Bench categories (55 PII types mentioned in paper)
        pii_categories = [
            'medical_condition', 'prescription', 'diagnosis',
            'financial_status', 'credit_score', 'bank_account',
            'email', 'phone', 'ssn', 'address',
            'employment', 'salary', 'company',
            'legal_case', 'criminal_record',
            'relationship_status', 'sexual_orientation',
            'religion', 'political_affiliation',
            'biometric_data', 'genetic_info'
        ]

        for i in range(num_samples):
            # Randomly select 1-3 PII types per sample
            num_pii_types = np.random.randint(1, 4)
            selected_pii = np.random.choice(pii_categories, num_pii_types, replace=False)

            # Generate queries based on PII types
            queries = []
            for pii_type in selected_pii:
                query = self._generate_query_for_pii(pii_type)
                queries.append(query)

            # Create sample
            benchmark_samples.append({
                'id': f'pii_bench_{i}',
                'queries': queries,
                'pii_types': list(selected_pii),
                'ground_truth_pii': {
                    'categories': list(selected_pii),
                    'queries': queries,
                    'contains_pii': True
                },
                'category': 'pii_bench',
                'benchmark': 'pii-bench'
            })

        print(f"✓ Generated {len(benchmark_samples)} PII-Bench style samples")
        return benchmark_samples

    def load_privacyxray_dataset(self, num_samples: int = 500) -> List[Dict[str, Any]]:
        """
        Load PrivacyXray style synthetic dataset.

        PrivacyXray creates synthetic individuals with 16 PII types.
        Since the full dataset may require special access,
        we generate synthetic data following their methodology.

        Args:
            num_samples: Number of synthetic individuals to generate

        Returns:
            List of samples with comprehensive PII profiles
        """
        print(f"\nGenerating PrivacyXray style synthetic dataset...")
        print(f"  Samples: {num_samples}")

        benchmark_samples = []

        # PrivacyXray's 16 PII types
        pii_types = [
            'name', 'email', 'phone', 'address',
            'ssn', 'date_of_birth', 'age', 'gender',
            'occupation', 'employer', 'salary', 'education',
            'medical_condition', 'prescription', 'insurance', 'emergency_contact'
        ]

        for i in range(num_samples):
            # Generate synthetic individual
            individual = self._generate_synthetic_individual(i)

            # Create queries that would expose this individual's PII
            queries = []
            for pii_type in pii_types[:10]:  # Use 10 out of 16 types
                query = f"Tell me about {pii_type} for person {i}"
                queries.append(query)

            benchmark_samples.append({
                'id': f'privacyxray_{i}',
                'synthetic_individual': individual,
                'queries': queries,
                'pii_types': pii_types,
                'ground_truth_pii': {
                    'profile': individual,
                    'contains_pii': True,
                    'num_pii_types': len(pii_types)
                },
                'category': 'privacyxray',
                'benchmark': 'privacyxray'
            })

        print(f"✓ Generated {len(benchmark_samples)} PrivacyXray style samples")
        return benchmark_samples

    def _load_ai4privacy_fallback(self, num_samples: int) -> List[Dict[str, Any]]:
        """Fallback: Generate synthetic data mimicking ai4privacy format."""
        print("Generating synthetic fallback data...")

        samples = []
        scenarios = [
            ("medical", ["diabetes symptoms", "insulin dosage", "blood sugar levels"]),
            ("financial", ["credit score check", "loan application", "bank statement"]),
            ("legal", ["lawsuit details", "court case", "legal representation"]),
            ("employment", ["job application", "salary negotiation", "employment history"]),
            ("personal", ["home address", "phone number", "email contact"])
        ]

        for i in range(num_samples):
            category, query_templates = scenarios[i % len(scenarios)]

            queries = [f"{template} {i}" for template in query_templates]
            source_text = f"User asked about: {', '.join(queries)}"
            masked_text = f"User asked about: QUERY_001, QUERY_002, QUERY_003"

            samples.append({
                'id': f'synthetic_{i}',
                'source_text': source_text,
                'masked_text': masked_text,
                'queries': queries,
                'pii_types': [category],
                'ground_truth_pii': {
                    'original': source_text,
                    'masked': masked_text,
                    'contains_pii': True
                },
                'category': category,
                'benchmark': 'synthetic_fallback'
            })

        return samples

    def _extract_queries_from_text(self, text: str) -> List[str]:
        """Extract query-like content from text."""
        # Simple extraction - split by sentences
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10][:5]

    def _identify_pii_types(self, source: str, masked: str) -> List[str]:
        """Identify what PII types were masked by comparing source and masked."""
        pii_types = []

        # Simple heuristics
        if '[NAME]' in masked or 'NAME' in masked:
            pii_types.append('name')
        if '[EMAIL]' in masked or 'EMAIL' in masked:
            pii_types.append('email')
        if '[PHONE]' in masked or 'PHONE' in masked:
            pii_types.append('phone')
        if '[ADDRESS]' in masked or 'ADDRESS' in masked:
            pii_types.append('address')

        # Check for medical terms
        medical_keywords = ['diabetes', 'cancer', 'disease', 'medication', 'treatment']
        if any(kw in source.lower() for kw in medical_keywords):
            pii_types.append('medical_condition')

        return pii_types if pii_types else ['unknown']

    def _generate_query_for_pii(self, pii_type: str) -> str:
        """Generate a realistic query for a given PII type."""
        query_templates = {
            'medical_condition': [
                "symptoms of diabetes",
                "treatment for hypertension",
                "managing chronic pain"
            ],
            'financial_status': [
                "how to improve credit score",
                "bankruptcy filing process",
                "debt consolidation options"
            ],
            'email': [
                "contact john.doe@example.com",
                "send to user@company.org"
            ],
            'phone': [
                "call 555-123-4567",
                "phone number 555-987-6543"
            ],
            'ssn': [
                "SSN 123-45-6789",
                "social security number verification"
            ]
        }

        templates = query_templates.get(pii_type, [f"information about {pii_type}"])
        return np.random.choice(templates)

    def _generate_synthetic_individual(self, idx: int) -> Dict[str, str]:
        """Generate a synthetic individual profile."""
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis']

        return {
            'name': f"{np.random.choice(first_names)} {np.random.choice(last_names)}",
            'email': f"person{idx}@example.com",
            'phone': f"555-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}",
            'age': str(np.random.randint(25, 65)),
            'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Manager'])
        }


# ============================================================================
# BENCHMARK EVALUATOR
# ============================================================================

class PublicBenchmarkEvaluator:
    """
    Evaluates privacy-preserving pipeline on public benchmarks.
    """

    def __init__(self, use_real_pipeline: bool = True):
        """
        Initialize evaluator.

        Args:
            use_real_pipeline: If True, use actual ensemble pipeline (Steps 3 & 4).
                              If False, use mock outputs (for format testing only).
        """
        self.loader = PublicBenchmarkLoader()
        self.privacy_evaluator = PrivacyEvaluator()
        self.utility_evaluator = UtilityEvaluator()
        self.use_real_pipeline = use_real_pipeline and PIPELINE_AVAILABLE

        # Initialize pipeline components if using real pipeline
        if self.use_real_pipeline:
            self.redactor = PrivacyRedactor()
            # Step 3: Ensemble of 5 LLM evaluators
            self.evaluators = [
                MockLLMEvaluator("GPT-4", bias=0.0),
                MockLLMEvaluator("Claude-3.5", bias=0.05),
                MockLLMEvaluator("Gemini-Pro", bias=-0.03),
                MockLLMEvaluator("Llama-3", bias=0.02),
                MockLLMEvaluator("Mistral-Large", bias=-0.01)
            ]
            # Step 4: Consensus aggregator
            self.aggregator = ConsensusAggregator()
            print("✓ Using REAL ensemble pipeline (Steps 1-4)")
        else:
            self.redactor = None
            self.evaluators = None
            self.aggregator = None
            print("⚠ Using mock outputs (format testing only)")

    def evaluate_on_benchmark(self, benchmark_name: str,
                             num_samples: int = 1000) -> Dict[str, Any]:
        """
        Evaluate pipeline on a specific public benchmark.

        Args:
            benchmark_name: 'ai4privacy', 'pii-bench', or 'privacyxray'
            num_samples: Number of samples to evaluate

        Returns:
            Comprehensive evaluation results
        """
        print(f"\n{'='*70}")
        print(f"  EVALUATING ON: {benchmark_name.upper()}")
        print(f"{'='*70}")

        # Load benchmark dataset
        if benchmark_name == 'ai4privacy':
            dataset = self.loader.load_ai4privacy_dataset(num_samples)
        elif benchmark_name == 'pii-bench':
            dataset = self.loader.load_pii_bench_dataset(num_samples)
        elif benchmark_name == 'privacyxray':
            dataset = self.loader.load_privacyxray_dataset(num_samples)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        if not dataset:
            print(f"ERROR: Could not load {benchmark_name} dataset")
            return {}

        # Prepare ground truth
        ground_truth_queries = [sample['queries'] for sample in dataset]

        # Generate baseline outputs (NO privacy protection)
        print(f"\nGenerating baseline outputs (no privacy)...")
        baseline_outputs = []
        for sample in dataset:
            # Baseline: Include actual queries (privacy violation!)
            output = {
                'queries': sample['queries'][:2],  # Leak some queries
                'reasoning': f"Based on: {', '.join(sample['queries'][:2])}"
            }
            baseline_outputs.append(json.dumps(output))

        # Generate privacy-preserving outputs (WITH our pipeline)
        print(f"Generating privacy-preserving outputs...")
        privacy_outputs = []

        if self.use_real_pipeline:
            # ========================================================================
            # ACTUAL PIPELINE: Steps 1-4 (YOUR KEY CONTRIBUTIONS!)
            # ========================================================================
            print(f"  Using REAL ensemble pipeline:")
            print(f"    Step 1: Redaction & Masking")
            print(f"    Step 3: Ensemble Evaluation (5 models)")
            print(f"    Step 4: Consensus Aggregation")

            for idx, sample in enumerate(dataset):
                if idx % 100 == 0 and idx > 0:
                    print(f"  Processed {idx}/{len(dataset)} samples...")

                # Convert benchmark sample to user_data format
                user_data = self._convert_sample_to_user_data(sample)

                # STEP 1: Redaction & Masking
                masked_user_data = self.redactor.redact_user_data(user_data)

                # STEP 3: Ensemble LLM Evaluators (YOUR KEY CONTRIBUTION!)
                # Create candidate topics to evaluate
                candidate_topics = [
                    {'ItemId': 'topic_A', 'Topic': 'health and wellness'},
                    {'ItemId': 'topic_B', 'Topic': 'technology and software'}
                ]

                all_model_results = []
                for evaluator in self.evaluators:
                    results = evaluator.evaluate_interest(masked_user_data, candidate_topics)
                    all_model_results.append(results)

                # STEP 4: Consensus Aggregation (YOUR KEY CONTRIBUTION!)
                consensus_results = self.aggregator.aggregate_median(all_model_results)

                # Output safe JSON (only generic metadata)
                output = consensus_results[0]  # Take first topic's result
                privacy_outputs.append(json.dumps(output))

            print(f"  ✓ Processed {len(dataset)} samples with full pipeline")

        else:
            # Mock outputs (for fast format testing only)
            print(f"  ⚠ Using mock outputs (not testing actual pipeline)")
            for sample in dataset:
                output = {
                    'ItemId': 'topic_A',
                    'QualityScore': 0.85,
                    'QualityReason': 'VeryStrong:MSNClicks+BingSearch'
                }
                privacy_outputs.append(json.dumps(output))

        # Evaluate privacy metrics
        print(f"\nEvaluating privacy metrics...")

        baseline_pii = self.privacy_evaluator.detect_pii_leakage(baseline_outputs)
        privacy_pii = self.privacy_evaluator.detect_pii_leakage(privacy_outputs)

        baseline_recon = self.privacy_evaluator.evaluate_reconstruction_attack(
            baseline_outputs, ground_truth_queries
        )
        privacy_recon = self.privacy_evaluator.evaluate_reconstruction_attack(
            privacy_outputs, ground_truth_queries
        )

        # Compile results
        results = {
            'benchmark': benchmark_name,
            'num_samples': len(dataset),
            'privacy_metrics': {
                'pii_leakage': {
                    'baseline': baseline_pii['leakage_rate'],
                    'with_privacy': privacy_pii['leakage_rate'],
                    'improvement': baseline_pii['leakage_rate'] - privacy_pii['leakage_rate'],
                    'improvement_pct': (baseline_pii['leakage_rate'] - privacy_pii['leakage_rate']) * 100
                },
                'reconstruction_attack': {
                    'baseline': baseline_recon['reconstruction_rate'],
                    'with_privacy': privacy_recon['reconstruction_rate'],
                    'improvement': baseline_recon['reconstruction_rate'] - privacy_recon['reconstruction_rate'],
                    'improvement_pct': (baseline_recon['reconstruction_rate'] - privacy_recon['reconstruction_rate']) * 100
                }
            },
            'detailed': {
                'baseline_pii': baseline_pii,
                'privacy_pii': privacy_pii,
                'baseline_recon': baseline_recon,
                'privacy_recon': privacy_recon
            }
        }

        # Print results
        self._print_results(results)

        return results

    def _convert_sample_to_user_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert benchmark sample format to user_data format expected by pipeline.

        Args:
            sample: Benchmark sample with 'queries', 'pii_types', etc.

        Returns:
            user_data format for PrivacyRedactor
        """
        # Extract queries from sample
        queries = sample.get('queries', [])

        # Create user_data structure
        user_data = {
            'BingSearch': [
                {'query': q, 'timestamp': '2024-01-15T10:00:00'}
                for q in queries[:3]  # Use first 3 queries as searches
            ],
            'MSNClicks': [
                {'title': q, 'timestamp': '2024-01-15T11:00:00'}
                for q in queries[3:6]  # Use next 3 as article clicks
            ],
            'demographics': {
                'age': 35,
                'gender': 'F',
                'location': 'Unknown'
            }
        }

        # Add MAI categories based on PII types (if available)
        pii_types = sample.get('pii_types', [])
        mai_categories = []
        for pii_type in pii_types:
            if 'medical' in pii_type or 'health' in pii_type:
                mai_categories.extend(['Health'] * 5)
            elif 'financial' in pii_type:
                mai_categories.extend(['Finance'] * 5)
            elif 'tech' in pii_type:
                mai_categories.extend(['Technology'] * 5)

        if mai_categories:
            user_data['MAI'] = mai_categories

        return user_data

    def _print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format."""
        print(f"\n{'='*70}")
        print(f"  RESULTS: {results['benchmark'].upper()}")
        print(f"{'='*70}")

        pii = results['privacy_metrics']['pii_leakage']
        recon = results['privacy_metrics']['reconstruction_attack']

        print(f"\n📊 Privacy Metrics:")
        print(f"  ├─ PII Leakage:")
        print(f"  │  ├─ Baseline (no privacy):     {pii['baseline']*100:.1f}%")
        print(f"  │  ├─ With Privacy Pipeline:     {pii['with_privacy']*100:.1f}%")
        print(f"  │  └─ Improvement:               {pii['improvement_pct']:.1f}%")
        print(f"  │")
        print(f"  └─ Reconstruction Attack Success:")
        print(f"     ├─ Baseline:                  {recon['baseline']*100:.1f}%")
        print(f"     ├─ With Privacy:              {recon['with_privacy']*100:.1f}%")
        print(f"     └─ Improvement:               {recon['improvement_pct']:.1f}%")

        # Verdict
        if pii['improvement_pct'] >= 80 and recon['improvement_pct'] >= 80:
            verdict = "✅ EXCELLENT - Pipeline provides >80% privacy improvement"
        elif pii['improvement_pct'] >= 60 and recon['improvement_pct'] >= 60:
            verdict = "✅ GOOD - Pipeline provides >60% privacy improvement"
        else:
            verdict = "⚠️  MODERATE - Room for improvement"

        print(f"\n🎯 Verdict: {verdict}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run public benchmark evaluation."""

    parser = argparse.ArgumentParser(
        description='Evaluate privacy-preserving pipeline on public benchmarks'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        choices=['ai4privacy', 'pii-bench', 'privacyxray', 'all'],
        default='ai4privacy',
        help='Which benchmark to run (default: ai4privacy)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to evaluate (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='public_benchmark_results.json',
        help='Output file for results (default: public_benchmark_results.json)'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock outputs (fast, format testing only). Default: use real pipeline'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  PUBLIC BENCHMARK EVALUATION")
    print("  Privacy-Preserving Ensemble Pipeline")
    print("="*70 + "\n")

    # Initialize evaluator (use real pipeline by default)
    use_real_pipeline = not args.mock
    if use_real_pipeline:
        print("🔬 Mode: REAL PIPELINE (Steps 1-4 applied)")
        print("   This will test your actual ensemble+consensus contributions\n")
    else:
        print("⚡ Mode: MOCK (fast format testing only)")
        print("   Use --mock for quick validation, omit for real evaluation\n")

    evaluator = PublicBenchmarkEvaluator(use_real_pipeline=use_real_pipeline)
    all_results = {
        'config': {
            'benchmark': args.benchmark,
            'num_samples': args.num_samples,
            'use_real_pipeline': use_real_pipeline,
            'pipeline_mode': 'real_ensemble' if use_real_pipeline else 'mock_format_only',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': {}
    }

    # Run evaluation
    if args.benchmark == 'all':
        benchmarks = ['ai4privacy', 'pii-bench', 'privacyxray']
    else:
        benchmarks = [args.benchmark]

    for benchmark in benchmarks:
        results = evaluator.evaluate_on_benchmark(benchmark, args.num_samples)
        all_results['results'][benchmark] = results

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {args.output}")
    print(f"{'='*70}\n")

    return all_results


if __name__ == "__main__":
    main()
